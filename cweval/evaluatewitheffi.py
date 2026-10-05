import csv
import json
import math
import os
import statistics
from collections import defaultdict
import multiprocessing as mp
import queue as queue_module
from dataclasses import dataclass
from typing import Dict, List, Optional
import fire
from natsort import natsorted
from cweval.commons import BENCHMARK_DIR, compile_all_in
from cweval.run_testswitheffi import TestFileResult, run_tests

FUNCTIONALITY_ARGS = [
    "-m",
    "functionality",
    "-k",
    "not _unsafe",
    "-q",
]

def _norm(path: str) -> str:
    return path.replace("\\", "/")

def _relative_test_path(file_path: str, root: str) -> str:
    file_abs = os.path.abspath(file_path)
    root_abs = os.path.abspath(root)
    return _norm(os.path.relpath(file_abs, root_abs))

def _res_all_to_rel_map(res_all: dict) -> Dict[str, dict]:
    ans = {}
    anchor = "generated_X/"
    for key, value in res_all.items():
        normalized = _norm(key)
        idx = normalized.find(anchor)
        if idx < 0:
            continue
        rel = normalized[idx + len(anchor) :]
        ans[rel] = value
    return ans

def _filename_to_lang(path: str) -> str:
    filename = os.path.splitext(os.path.basename(path))[0]
    parts = filename.split("_")
    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate in {"c", "cpp", "go", "js", "py"}:
            return candidate
    return "py"

def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))

def _geomean(values: List[float]) -> Optional[float]:
    values = [x for x in values if x > 0]
    if not values:
        return None
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))
@dataclass
class TimingFileResult:
    file: str
    functional_runtime: Optional[float]

def _timing_worker(
    return_queue,
    test_path: str,
    timeout_per_test: float,
) -> None:
    try:
        results = run_tests(
            test_path,
            timeout_per_test=timeout_per_test,
            args=FUNCTIONALITY_ARGS,
        )
        compact = [
            TimingFileResult(
                file=r.file,
                functional_runtime=r.functional_runtime,
            )
            for r in results
        ]
        return_queue.put(("ok", compact))
    except BaseException as exc:
        return_queue.put(
            (
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )

def _run_once(
    test_path: str,
    timeout_per_test: float,
) -> List[TimingFileResult]:
    return_queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_timing_worker,
        args=(return_queue, test_path, timeout_per_test),
    )
    process.start()

    # Read while the child is alive. Do not join first.
    while True:
        try:
            status, payload = return_queue.get(timeout=1.0)
            break
        except queue_module.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"Timing subprocess exited before returning a result "
                    f"(exitcode={process.exitcode}, test_path={test_path})"
                )
    process.join()
    return_queue.close()
    return_queue.join_thread()
    if status == "error":
        raise RuntimeError(
            f"Timing subprocess failed for {test_path}: {payload}"
        )
    return payload


def _measure_path(
    test_path: str,
    warmup: int,
    repeat: int,
    timeout_per_test: float,
    lang: str = "",
) -> Dict[str, dict]:
    print("\n" + "=" * 80)
    print(f"Timing path: {test_path}")
    print(
        f"warmup={warmup}, repeat={repeat}, "
        f"timeout_per_test={timeout_per_test}, lang={lang or 'all'}"
    )
    print("=" * 80, flush=True)
    for i in range(warmup):
        print(f"[warmup {i + 1}/{warmup}] {test_path}", flush=True)
        _run_once(test_path, timeout_per_test)
    runtimes: Dict[str, List[float]] = defaultdict(list)
    for i in range(repeat):
        print(f"[measurement {i + 1}/{repeat}] {test_path}", flush=True)
        results = _run_once(test_path, timeout_per_test)
        for file_result in results:
            rel = _relative_test_path(file_result.file, test_path)
            if lang and _filename_to_lang(rel) != lang:
                continue
            runtime = file_result.functional_runtime
            if runtime is not None and runtime > 0:
                runtimes[rel].append(float(runtime))
    measured = {}
    for rel, values in runtimes.items():
        measured[rel] = {
            "runtime_s": _median(values),
            "runs_s": values,
            "successful_runs": len(values),
            "requested_runs": repeat,
        }
    print(
        f"[timing] {test_path}: collected runtime for "
        f"{len(measured)} task files",
        flush=True,
    )
    if not measured:
        print(
            f"[timing][WARNING] No valid functionality runtimes from "
            f"{test_path}. Check pytest failures/collection errors above.",
            flush=True,
        )
    return measured

def _sample_index(generated_path: str) -> int:
    name = os.path.basename(os.path.normpath(generated_path))
    if not name.startswith("generated_"):
        raise ValueError(f"Invalid generated path: {generated_path}")
    return int(name.split("_")[-1])

def _is_eligible(
    res: dict,
    sample_idx: int,
    require_secure: bool,
) -> bool:
    functional = res.get("functional", [])
    secure = res.get("secure", [])
    func_secure = res.get("func_secure", [])
    if sample_idx >= len(functional):
        return False
    if require_secure:
        if sample_idx < len(func_secure):
            return bool(func_secure[sample_idx])
        if sample_idx < len(secure):
            return bool(functional[sample_idx] and secure[sample_idx])
        return False
    return bool(functional[sample_idx])


def _sample_status(res: dict, sample_idx: int) -> dict:
    functional_list = res.get("functional", [])
    secure_list = res.get("secure", [])
    func_secure_list = res.get("func_secure", [])
    functional = (
        bool(functional_list[sample_idx])
        if sample_idx < len(functional_list)
        else False
    )
    secure = (
        bool(secure_list[sample_idx])
        if sample_idx < len(secure_list)
        else False
    )
    func_secure = (
        bool(func_secure_list[sample_idx])
        if sample_idx < len(func_secure_list)
        else functional and secure
    )
    return {
        "functional": functional,
        "secure": secure,
        "func_secure": func_secure,
    }


def _summarize(
    rows: List[dict],
    require_secure: bool,
    faster_threshold: float,
) -> dict:
    if require_secure:
        eligible = [r for r in rows if r["func_secure"]]
        group = "FS"
    else:
        eligible = [r for r in rows if r["functional"]]
        group = "F"
    measured = [
        r
        for r in eligible
        if r["generated_runtime_s"] is not None
        and r["reference_runtime_s"] is not None
        and r["speedup"] is not None
        and r["speedup"] > 0
    ]
    speedups = [r["speedup"] for r in measured]
    faster = [s for s in speedups if s > faster_threshold]
    return {
        "group": group,
        "num_eligible": len(eligible),
        "num_measured": len(measured),
        "measurement_coverage_percent": (
            len(measured) / len(eligible) * 100 if eligible else 0.0
        ),
        "ER_percent": (
            len(faster) / len(measured) * 100 if measured else 0.0
        ),
        # Keep arithmetic mean for compatibility with many existing
        # code-efficiency experiments.
        "AS_arithmetic": (
            float(statistics.mean(speedups)) if speedups else None
        ),
        # Also report geometric mean, which is usually preferable when
        # averaging multiplicative ratios such as speedups.
        "AS_geomean": _geomean(speedups),
        "median_speedup": _median(speedups),
        "faster_threshold": faster_threshold,
    }


class EfficiencyEvaler:
    def __init__(
        self,
        eval_path: str,
        warmup: int = 3,
        repeat: int = 20,
        timeout_per_test: float = 3.0,
        faster_threshold: float = 1.0,
        lang: str = "",
    ):
        self.eval_path = os.path.normpath(eval_path)
        self.warmup = int(warmup)
        self.repeat = int(repeat)
        self.timeout_per_test = float(timeout_per_test)
        self.faster_threshold = float(faster_threshold)
        self.lang = str(lang).strip().lower()

        if self.lang and self.lang not in {"c", "cpp", "go", "py", "js"}:
            raise ValueError(
                f"Unsupported lang={self.lang!r}; "
                "expected c/cpp/go/py/js or empty string."
            )
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeat <= 0:
            raise ValueError("repeat must be > 0")
        if self.faster_threshold <= 0:
            raise ValueError("faster_threshold must be > 0")
        res_all_path = os.path.join(self.eval_path, "res_all.json")
        if not os.path.isfile(res_all_path):
            raise FileNotFoundError(
                f"{res_all_path} does not exist. "
                "Run cweval/evaluate.py pipeline first."
            )
        with open(res_all_path, "r") as f:
            self.res_all = json.load(f)
        self.res_by_rel = _res_all_to_rel_map(self.res_all)
        self.generated_paths = [
            os.path.join(self.eval_path, name)
            for name in natsorted(os.listdir(self.eval_path))
            if name.startswith("generated_")
            and os.path.isdir(os.path.join(self.eval_path, name))
        ]
        if not self.generated_paths:
            raise RuntimeError(
                f"No generated_* directories found under {self.eval_path}"
            )

    def run(self) -> None:
        print("\nCWEval efficiency evaluation")
        print(f"eval_path          : {self.eval_path}")
        print(f"generated samples  : {len(self.generated_paths)}")
        print(f"warmup             : {self.warmup}")
        print(f"repeat             : {self.repeat}")
        print(f"timeout/test       : {self.timeout_per_test}s")
        print(f"faster threshold   : {self.faster_threshold}x")
        print(f"language           : {self.lang or 'all'}")
        print(
            "\nIMPORTANT: run this with one evaluation process only; "
            "do not run several timing jobs in parallel.\n"
        )
        print("Compiling reference benchmark (outside timed region)...", flush=True)
        compile_all_in(BENCHMARK_DIR, check=False, num_proc=1)
        reference_measurements = _measure_path(
            BENCHMARK_DIR,
            warmup=self.warmup,
            repeat=self.repeat,
            timeout_per_test=self.timeout_per_test,
            lang=self.lang,
        )
        generated_measurements: Dict[int, Dict[str, dict]] = {}
        for generated_path in self.generated_paths:
            sample_idx = _sample_index(generated_path)
            generated_measurements[sample_idx] = _measure_path(
                generated_path,
                warmup=self.warmup,
                repeat=self.repeat,
                timeout_per_test=self.timeout_per_test,
                lang=self.lang,
            )
        rows: List[dict] = []
        tasks: Dict[str, dict] = {}
        for rel, cweval_res in sorted(self.res_by_rel.items()):
            if self.lang and _filename_to_lang(rel) != self.lang:
                continue
            ref_info = reference_measurements.get(rel)
            ref_runtime = ref_info["runtime_s"] if ref_info else None
            task_samples = []
            for generated_path in self.generated_paths:
                sample_idx = _sample_index(generated_path)
                status = _sample_status(cweval_res, sample_idx)
                gen_info = generated_measurements.get(sample_idx, {}).get(rel)
                gen_runtime = gen_info["runtime_s"] if gen_info else None
                speedup = None
                if (
                    ref_runtime is not None
                    and gen_runtime is not None
                    and ref_runtime > 0
                    and gen_runtime > 0
                ):
                    speedup = ref_runtime / gen_runtime
                row = {
                    "task": rel,
                    "lang": _filename_to_lang(rel),
                    "sample_idx": sample_idx,
                    **status,
                    "reference_runtime_s": ref_runtime,
                    "generated_runtime_s": gen_runtime,
                    "speedup": speedup,
                    "reference_successful_runs": (
                        ref_info["successful_runs"] if ref_info else 0
                    ),
                    "generated_successful_runs": (
                        gen_info["successful_runs"] if gen_info else 0
                    ),
                }
                rows.append(row)
                task_samples.append(row.copy())
            tasks[rel] = {
                "lang": _filename_to_lang(rel),
                "reference_runtime_s": ref_runtime,
                "reference_runs_s": (
                    ref_info["runs_s"] if ref_info else []
                ),
                "samples": task_samples,
            }
        summary_f = _summarize(
            rows,
            require_secure=False,
            faster_threshold=self.faster_threshold,
        )
        summary_fs = _summarize(
            rows,
            require_secure=True,
            faster_threshold=self.faster_threshold,
        )
        output = {
            "metadata": {
                "eval_path": self.eval_path,
                "benchmark_dir": BENCHMARK_DIR,
                "warmup": self.warmup,
                "repeat": self.repeat,
                "timeout_per_test": self.timeout_per_test,
                "faster_threshold": self.faster_threshold,
                "language": self.lang or "all",
                "runtime_definition": (
                    "sum of pytest call-phase durations of all selected "
                    "functionality tests for one CWEval task; median across "
                    "repeated benchmark runs"
                ),
                "speedup_definition": (
                    "official_reference_runtime / generated_runtime"
                ),
            },
            "summary": {
                "functional": summary_f,
                "functional_secure": summary_fs,
            },
            "tasks": tasks,
        }
        json_path = os.path.join(
            self.eval_path, "efficiency_results.json"
        )
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        csv_path = os.path.join(
            self.eval_path, "efficiency_details.csv"
        )
        fieldnames = [
            "task",
            "lang",
            "sample_idx",
            "functional",
            "secure",
            "func_secure",
            "reference_runtime_s",
            "generated_runtime_s",
            "speedup",
            "reference_successful_runs",
            "generated_successful_runs",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + "=" * 80)
        print("Efficiency summary")
        print("=" * 80)
        for label, summary in [
            ("F  (functional)", summary_f),
            ("FS (functional & secure)", summary_fs),
        ]:
            print(f"\n{label}")
            print(f"eligible: {summary['num_eligible']}")
            print(f"measured: {summary['num_measured']}")
            print(
                "coverage: "
                f"{summary['measurement_coverage_percent']:.2f}%"
            )
            print(f"ER: {summary['ER_percent']:.2f}%")
            if summary["AS_arithmetic"] is not None:
                print(
                    " AS arithmetic: "
                    f"{summary['AS_arithmetic']:.4f}x"
                )
            if summary["AS_geomean"] is not None:
                print(
                    "AS geomean: "
                    f"{summary['AS_geomean']:.4f}x"
                )
            if summary["median_speedup"] is not None:
                print(
                    "median speedup: "
                    f"{summary['median_speedup']:.4f}x"
                )
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
if __name__ == "__main__":
    fire.Fire(EfficiencyEvaler)