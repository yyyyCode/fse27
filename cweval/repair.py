import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import fire
import torch
from natsort import natsorted
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None

def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")

def _res_all_to_rel_map(res_all: dict) -> Dict[str, dict]:
    ans = {}
    for key, value in res_all.items():
        normalized = _norm(key)
        anchor = "generated_X/"
        idx = normalized.find(anchor)
        if idx >= 0:
            rel = normalized[idx + len(anchor):]
        else:
            parts = normalized.split("/")
            generated_pos = next(
                (
                    i for i, p in enumerate(parts)
                    if p.startswith("generated_")
                ),
                None,
            )
            if generated_pos is not None:
                rel = "/".join(parts[generated_pos + 1:])
            else:
                rel = normalized
        ans[rel] = value
    return ans

class CSERepairer:
    begin_prompt_anchor = "BEGIN PROMPT"
    begin_solution_anchor = "BEGIN SOLUTION"
    def __init__(
        self,
        input_eval_path: str,
        output_eval_path: str,
        model: str = "",
        ppt: str = "direct",
        langs: List[str] = LANGS,
        n_samples: int = 10,
        faster_threshold: float = 1.0,
        gpu_id: int = 0,
        max_completion_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int = 42,
        repair_all: bool = False,
        use_feedback: bool = True,
        include_numeric_speedup: bool = False,
        overwrite: bool = False,
    ):
        self.input_eval_path = os.path.normpath(input_eval_path)
        self.output_eval_path = os.path.normpath(output_eval_path)
        self.model_path = model
        self.ppt = ppt
        self.langs = list(langs)
        self.n_samples = int(n_samples)
        self.faster_threshold = float(faster_threshold)
        self.gpu_id = int(gpu_id)
        self.max_completion_tokens = int(max_completion_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.repair_all = _as_bool(repair_all)
        self.use_feedback = _as_bool(use_feedback)
        self.include_numeric_speedup = _as_bool(include_numeric_speedup)
        self.overwrite = _as_bool(overwrite)
        self.do_sample = self.temperature > 0
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.faster_threshold <= 0:
            raise ValueError("faster_threshold must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0 < self.top_p <= 1.0):
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        self._validate_inputs()
        self.res_by_rel = self._load_res_all()
        self.eff_by_key = self._load_efficiency_details()
        self.cases = self._get_cases()
        self._load_model()
    def _validate_inputs(self) -> None:
        if not os.path.isdir(self.input_eval_path):
            raise FileNotFoundError(self.input_eval_path)
        expected = [
            os.path.join(self.input_eval_path, f"generated_{i}")
            for i in range(self.n_samples)
        ]
        missing = [p for p in expected if not os.path.isdir(p)]
        if missing:
            raise RuntimeError(
                "Missing sampling directories:\n"
                + "\n".join(missing)
            )
        res_path = os.path.join(self.input_eval_path, "res_all.json")
        eff_path = os.path.join(
            self.input_eval_path,
            "efficiency_details.csv",
        )
        if not os.path.isfile(res_path):
            raise FileNotFoundError(
                f"{res_path} not found. Run evaluate_sampling.py first."
            )
        if not os.path.isfile(eff_path):
            raise FileNotFoundError(
                f"{eff_path} not found. Run "
                "evaluatewitheffi_sampling.py first."
            )
        if os.path.abspath(self.input_eval_path) == os.path.abspath(
            self.output_eval_path
        ):
            raise ValueError(
                "output_eval_path must differ from input_eval_path."
            )
        os.makedirs(self.output_eval_path, exist_ok=True)
    def _load_res_all(self) -> Dict[str, dict]:
        path = os.path.join(self.input_eval_path, "res_all.json")
        with open(path, "r") as f:
            res_all = json.load(f)
        rel_map = _res_all_to_rel_map(res_all)
        bad = []
        for rel, item in rel_map.items():
            for key in ("functional", "secure", "func_secure"):
                values = item.get(key, [])
                if len(values) != self.n_samples:
                    bad.append((rel, key, len(values)))
        if bad:
            preview = "\n".join(
                f"{r}: {k} has {n}"
                for r, k, n in bad[:10]
            )
            raise RuntimeError(
                "res_all.json is not aligned with n_samples="
                f"{self.n_samples}.\n{preview}"
            )
        return rel_map
    def _load_efficiency_details(
        self,
    ) -> Dict[Tuple[str, int], dict]:
        path = os.path.join(
            self.input_eval_path,
            "efficiency_details.csv",
        )
        ans = {}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"task", "sample_idx", "speedup"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise RuntimeError(
                    "efficiency_details.csv must contain "
                    "task, sample_idx, and speedup columns."
                )
            for row in reader:
                rel = _norm(row["task"])
                sample_idx = int(row["sample_idx"])
                ans[(rel, sample_idx)] = {
                    "speedup": _to_float(row.get("speedup")),
                    "functional": _as_bool(row.get("functional")),
                    "secure": _as_bool(row.get("secure")),
                    "func_secure": _as_bool(row.get("func_secure")),
                }
        return ans
    def _load_model(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        gpu_count = torch.cuda.device_count()
        if self.gpu_id < 0 or self.gpu_id >= gpu_count:
            raise ValueError(
                f"Invalid gpu_id={self.gpu_id}; available 0-{gpu_count - 1}"
            )
        self.device = torch.device(f"cuda:{self.gpu_id}")
        print(f"Using model: {self.model_path}")
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(self.gpu_id)}")
        print(
            "Repair decoding: "
            + (
                f"sampling temperature={self.temperature}, top_p={self.top_p}"
                if self.do_sample
                else "greedy"
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
    def _get_cases(self) -> List[dict]:
        cases = []
        for root, _, files in os.walk(BENCHMARK_DIR):
            if "__pycache__" in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                lang = ext[1:]
                if not ext or not file_wo_ext.endswith("_task"):
                    continue
                if lang not in self.langs:
                    continue
                task_file_path = os.path.join(root, file)
                with open(task_file_path, "r") as f:
                    task_code = f.read()
                begin_solution_line = ""
                for line in task_code.splitlines():
                    if self.begin_solution_anchor in line:
                        begin_solution_line = line
                        break
                if not begin_solution_line:
                    raise ValueError(
                        f"No BEGIN SOLUTION anchor in {task_file_path}"
                    )
                code_prompt = (
                    task_code
                    .split(self.begin_prompt_anchor)[-1]
                    .split(begin_solution_line)[0]
                    .strip()
                )
                rel_task = _norm(
                    os.path.relpath(task_file_path, BENCHMARK_DIR)
                )
                rel_raw = rel_task.replace("_task.", "_raw.")
                rel_test = rel_task.replace("_task.", "_test.")
                cases.append(
                    {
                        "task_file_path": task_file_path,
                        "rel_task": rel_task,
                        "rel_raw": rel_raw,
                        "rel_test": rel_test,
                        "lang": lang,
                        "code_prompt": code_prompt,
                    }
                )
        return cases
    def _official_prompt(self, case: dict) -> str:
        prompt = make_prompt(self.ppt)
        return prompt.PPT.format(
            lang=case["lang"],
            lang_instr=prompt.LANG_INSTR[case["lang"]],
            code_prompt=case["code_prompt"],
        )
    def _status(self, case: dict, sample_idx: int) -> dict:
        rel_test = case["rel_test"]
        result = self.res_by_rel.get(rel_test, {})
        f_list = result.get("functional", [])
        s_list = result.get("secure", [])
        fs_list = result.get("func_secure", [])
        functional = (
            _as_bool(f_list[sample_idx])
            if sample_idx < len(f_list)
            else False
        )
        secure = (
            _as_bool(s_list[sample_idx])
            if sample_idx < len(s_list)
            else False
        )
        func_secure = (
            _as_bool(fs_list[sample_idx])
            if sample_idx < len(fs_list)
            else (functional and secure)
        )
        eff = self.eff_by_key.get((rel_test, sample_idx), {})
        speedup = _to_float(eff.get("speedup"))

        efficient = (
            speedup is not None
            and speedup > self.faster_threshold
        )
        cse = func_secure and efficient
        return {
            "functional": functional,
            "secure": secure,
            "func_secure": func_secure,
            "speedup": speedup,
            "efficient": efficient,
            "cse": cse,
        }
    def _feedback_text(self, status: dict) -> str:
        lines = [
            "Coarse evaluation feedback:",
            "- Functional correctness: "
            + ("PASS" if status["functional"] else "FAIL"),
            "- Security: "
            + ("PASS" if status["secure"] else "FAIL"),
            "- Efficiency: "
            + ("PASS" if status["efficient"] else "FAIL"),
        ]
        if (
            self.include_numeric_speedup
            and status["speedup"] is not None
        ):
            lines.append(
                "- Measured runtime speedup vs. the benchmark reference: "
                f"{status['speedup']:.4f}x "
                f"(required > {self.faster_threshold:.4f}x)."
            )
        else:
            lines.append(
                "- Efficiency requirement: runtime speedup must be "
                f"> {self.faster_threshold:.4f}x."
            )
        return "\n".join(lines)
    def _build_repair_prompt(
        self,
        case: dict,
        candidate_code: str,
        status: dict,
    ) -> str:
        return (
            "You are repairing a candidate solution for a code-generation "
            "benchmark.\n\n"
            "Goal: produce a replacement implementation that simultaneously "
            "satisfies functional correctness, security, and execution "
            "efficiency.\n\n"
            "Rules:\n"
            "1. Preserve the required interface and intended behavior.\n"
            "2. Fix correctness problems and edge cases.\n"
            "3. Use secure coding practices: validate inputs/bounds where "
            "applicable, avoid unsafe APIs/operations, memory-safety bugs, "
            "injection risks, insecure resource handling, and related "
            "vulnerabilities.\n"
            "4. Improve runtime/memory efficiency where possible; avoid "
            "unnecessary work, repeated computation, allocations, copies, "
            "and inefficient algorithms/data structures.\n"
            "5. Do not remove required functionality merely to pass security "
            "checks or improve speed.\n"
            "6. Return ONLY the complete replacement source code. Do not use "
            "Markdown fences and do not explain the changes.\n\n"
            + (
                self._feedback_text(status)
                if self.use_feedback
                else (
                    "No evaluator feedback is provided. Independently inspect "
                    "the candidate for correctness, security, and efficiency."
                )
            )
            + "\n\n"
            "Original task specification:\n"
            "---------------- ORIGINAL TASK ----------------\n"
            + self._official_prompt(case)
            + "\n---------------- END ORIGINAL TASK ------------\n\n"
            "Candidate implementation to repair:\n"
            f"---------------- CANDIDATE ({case['lang']}) ----------------\n"
            + candidate_code
            + "\n---------------- END CANDIDATE ----------------\n\n"
            "Return the repaired complete source code now."
        )
    @staticmethod
    def _extract_code(response: str) -> str:
        response = response.strip()
        match = re.search(
            r"```[^\n]*\n(.*?)```",
            response,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if response.startswith("```"):
            lines = response.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return response
    def _repair_seed(self, case: dict, sample_idx: int) -> int:
        key = (
            f"{case['rel_task']}::{sample_idx}::repair"
        ).encode("utf-8")
        h = int(hashlib.md5(key).hexdigest()[:8], 16)
        return (self.seed + h) % 2_147_483_647
    def _generate(self, prompt_text: str, seed: int) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        if self.tokenizer.chat_template is not None:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = prompt_text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = {
            "max_new_tokens": self.max_completion_tokens,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            kwargs["do_sample"] = False
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **kwargs,
            )
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return self._extract_code(response)
    def _source_path(self, case: dict, sample_idx: int) -> str:
        return os.path.join(
            self.input_eval_path,
            f"generated_{sample_idx}",
            case["rel_raw"],
        )
    def _output_path(self, case: dict, sample_idx: int) -> str:
        return os.path.join(
            self.output_eval_path,
            f"generated_{sample_idx}",
            case["rel_raw"],
        )
    def _copy_or_repair(
        self,
        case: dict,
        sample_idx: int,
    ) -> dict:
        src = self._source_path(case, sample_idx)
        dst = self._output_path(case, sample_idx)
        if not os.path.isfile(src):
            return {
                "task": case["rel_test"],
                "raw_file": case["rel_raw"],
                "sample_idx": sample_idx,
                "action": "missing_input",
                "output": dst,
            }
        status = self._status(case, sample_idx)
        record = {
            "task": case["rel_test"],
            "raw_file": case["rel_raw"],
            "lang": case["lang"],
            "sample_idx": sample_idx,
            "before_functional": status["functional"],
            "before_secure": status["secure"],
            "before_func_secure": status["func_secure"],
            "before_speedup": status["speedup"],
            "before_efficient": status["efficient"],
            "before_cse": status["cse"],
            "threshold": self.faster_threshold,
        }
        if os.path.isfile(dst) and not self.overwrite:
            record["action"] = "skip_existing"
            record["output"] = dst
            return record
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if status["cse"] and not self.repair_all:
            shutil.copy2(src, dst)
            record["action"] = "copy_cse"
            record["output"] = dst
            return record
        with open(src, "r") as f:
            candidate_code = f.read()
        prompt_text = self._build_repair_prompt(
            case,
            candidate_code,
            status,
        )
        repaired_code = self._generate(
            prompt_text,
            seed=self._repair_seed(case, sample_idx),
        )
        if not repaired_code.strip():
            # Conservative fallback: never lose the original candidate.
            shutil.copy2(src, dst)
            record["action"] = "repair_empty_fallback_copy"
        else:
            with open(dst, "w") as f:
                f.write(repaired_code)
            record["action"] = "repair"
        record["output"] = dst
        return record
    def run(self) -> None:
        print("=" * 80)
        print("RQ5 CWEval CSE Repair")
        print("=" * 80)
        print(f"input : {self.input_eval_path}")
        print(f"output: {self.output_eval_path}")
        print(f"samples: {self.n_samples}")
        print(f"CSE efficiency threshold: > {self.faster_threshold}x")
        print(f"repair_all: {self.repair_all}")
        print(
            "Policy: already-CSE candidates are copied unchanged; "
            "other candidates are repaired once."
        )
        records = []
        total = len(self.cases) * self.n_samples
        with tqdm(total=total, desc="repair") as pbar:
            for case in self.cases:
                for sample_idx in range(self.n_samples):
                    try:
                        record = self._copy_or_repair(
                            case,
                            sample_idx,
                        )
                    except Exception as e:
                        record = {
                            "task": case["rel_test"],
                            "raw_file": case["rel_raw"],
                            "lang": case["lang"],
                            "sample_idx": sample_idx,
                            "action": "error",
                            "error": repr(e),
                        }
                        print(
                            f"\n[ERROR] {case['rel_test']} "
                            f"sample={sample_idx}: {e}",
                            flush=True,
                        )

                    records.append(record)
                    pbar.update(1)
        json_path = os.path.join(
            self.output_eval_path,
            "repair_manifest.json",
        )
        with open(json_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "input_eval_path": self.input_eval_path,
                        "output_eval_path": self.output_eval_path,
                        "model": self.model_path,
                        "n_samples": self.n_samples,
                        "faster_threshold": self.faster_threshold,
                        "repair_all": self.repair_all,
                        "use_feedback": self.use_feedback,
                        "include_numeric_speedup":
                            self.include_numeric_speedup,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "seed": self.seed,
                        "feedback_policy": (
                            (
                                "Only coarse pass/fail C/S/E feedback and "
                                "optional aggregate speedup are exposed. "
                                "No benchmark tests, reference solution, CWE "
                                "label, or detailed failure trace is exposed."
                            )
                            if self.use_feedback
                            else (
                                "No evaluator feedback is exposed to the model."
                            )
                        ),
                    },
                    "records": records,
                },
                f,
                indent=2,
            )
        csv_path = os.path.join(
            self.output_eval_path,
            "repair_manifest.csv",
        )
        fieldnames = sorted(
            set().union(*(r.keys() for r in records))
        ) if records else []
        if fieldnames:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(records)
        counts = defaultdict(int)
        for r in records:
            counts[r.get("action", "unknown")] += 1
        print("\nRepair actions:")
        for action, count in sorted(counts.items()):
            print(f"  {action:<28} {count}")
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
        print(
            "\nNext:\n"
            "1) evaluate_sampling.py pipeline on the output path\n"
            "2) evaluatewitheffi_sampling.py run on the output path\n"
            "3) compare CSE@k before vs after repair"
        )
if __name__ == "__main__":
    fire.Fire(CSERepairer)