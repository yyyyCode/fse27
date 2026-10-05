import csv
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional
import fire
from cweval.evaluatewitheffi import EfficiencyEvaler

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return bool(value)

def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x

def _mean(values: List[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None

def _std(values: List[float]) -> Optional[float]:
    return (
        float(statistics.pstdev(values))
        if len(values) > 1
        else 0.0 if values else None
    )

def _geomean(values: List[float]) -> Optional[float]:
    values = [x for x in values if x is not None and x > 0]
    if not values:
        return None
    return float(
        math.exp(sum(math.log(x) for x in values) / len(values))
    )

def _summarize_speedups(
    rows: List[dict],
    eligible_key: str,
    faster_threshold: float,
) -> dict:
    eligible = [r for r in rows if r[eligible_key]]
    measured = [
        r
        for r in eligible
        if r["speedup"] is not None
        and r["speedup"] > 0
    ]
    speedups = [r["speedup"] for r in measured]
    faster = [
        x for x in speedups
        if x > faster_threshold
    ]
    return {
        "eligible": len(eligible),
        "measured": len(measured),
        "coverage_percent": (
            len(measured) / len(eligible) * 100.0
            if eligible else 0.0
        ),
        "ER_percent": (
            len(faster) / len(measured) * 100.0
            if measured else 0.0
        ),
        "AS_arithmetic": _mean(speedups),
        "AS_geomean": _geomean(speedups),
        "median_speedup": (
            float(statistics.median(speedups))
            if speedups else None
        ),
    }

class SamplingEfficiencyEvaler(EfficiencyEvaler):
    def __init__(
        self,
        eval_path: str,
        warmup: int = 3,
        repeat: int = 20,
        timeout_per_test: float = 3.0,
        faster_threshold: float = 1.0,
        lang: str = "",
        n_samples: int = 10,
        strict_samples: bool = True,
    ):
        super().__init__(
            eval_path=eval_path,
            warmup=warmup,
            repeat=repeat,
            timeout_per_test=timeout_per_test,
            faster_threshold=faster_threshold,
            lang=lang,
        )
        self.n_samples = int(n_samples)
        self.strict_samples = _as_bool(strict_samples)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        self._validate_generated_samples()
    def _validate_generated_samples(self) -> None:
        found = []
        for path in self.generated_paths:
            name = os.path.basename(os.path.normpath(path))
            suffix = name[len("generated_") :]

            if not suffix.isdigit():
                raise ValueError(
                    f"Invalid generated directory name: {name}"
                )
            found.append(int(suffix))
        expected = list(range(self.n_samples))
        if self.strict_samples and found != expected:
            raise RuntimeError(
                "Sampling directories are not complete/contiguous.\n"
                f"Expected: {expected}\n"
                f"Found   : {found}"
            )
        if not self.strict_samples:
            allowed = set(expected)
            self.generated_paths = [
                p
                for p in self.generated_paths
                if int(os.path.basename(p).split("_")[-1]) in allowed
            ]
        if self.strict_samples:
            for _, res in self.res_all.items():
                for key in ("functional", "secure", "func_secure"):
                    values = res.get(key, [])
                    if len(values) != self.n_samples:
                        raise RuntimeError(
                            "res_all.json sample count mismatch: "
                            f"{key} has {len(values)} values, "
                            f"expected {self.n_samples}. "
                            "Run evaluate_sampling.py first."
                        )
    def _load_detail_rows(self) -> List[dict]:
        csv_path = os.path.join(
            self.eval_path,
            "efficiency_details.csv",
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} does not exist. "
                "The base efficiency evaluation did not finish correctly."
            )
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = dict(raw)
                row["sample_idx"] = int(row["sample_idx"])
                row["functional"] = _to_bool(row["functional"])
                row["secure"] = _to_bool(row["secure"])
                row["func_secure"] = _to_bool(row["func_secure"])
                for key in (
                    "reference_runtime_s",
                    "generated_runtime_s",
                    "speedup",
                ):
                    row[key] = _to_float(row.get(key))
                rows.append(row)
        return rows
    def report_sampling_efficiency(self) -> dict:
        rows = self._load_detail_rows()
        by_sample: Dict[int, List[dict]] = defaultdict(list)
        for row in rows:
            by_sample[row["sample_idx"]].append(row)
        per_sample = []
        csv_rows = []
        for sample_idx in range(self.n_samples):
            sample_rows = by_sample.get(sample_idx, [])
            if self.strict_samples and not sample_rows:
                raise RuntimeError(
                    f"No timing rows found for sample {sample_idx}"
                )
            n_tasks = len(sample_rows)
            functional_count = sum(
                int(r["functional"]) for r in sample_rows
            )
            secure_count = sum(
                int(r["secure"]) for r in sample_rows
            )
            func_secure_count = sum(
                int(r["func_secure"]) for r in sample_rows
            )
            summary_f = _summarize_speedups(
                sample_rows,
                eligible_key="functional",
                faster_threshold=self.faster_threshold,
            )
            summary_fs = _summarize_speedups(
                sample_rows,
                eligible_key="func_secure",
                faster_threshold=self.faster_threshold,
            )
            cse_count = sum(
                1
                for r in sample_rows
                if (
                    r["func_secure"]
                    and r["speedup"] is not None
                    and r["speedup"] > self.faster_threshold
                )
            )
            def pct(x: int) -> float:
                return x / n_tasks * 100.0 if n_tasks else 0.0
            item = {
                "sample_idx": sample_idx,
                "num_tasks": n_tasks,
                "C_percent": pct(functional_count),
                "S_percent": pct(secure_count),
                "CS_percent": pct(func_secure_count),
                "ER_F_percent": summary_f["ER_percent"],
                "ER_FS_percent": summary_fs["ER_percent"],
                "AS_F_arithmetic": summary_f["AS_arithmetic"],
                "AS_F_geomean": summary_f["AS_geomean"],
                "AS_FS_arithmetic": summary_fs["AS_arithmetic"],
                "AS_FS_geomean": summary_fs["AS_geomean"],
                "F_eligible": summary_f["eligible"],
                "F_measured": summary_f["measured"],
                "F_coverage_percent": summary_f["coverage_percent"],
                "FS_eligible": summary_fs["eligible"],
                "FS_measured": summary_fs["measured"],
                "FS_coverage_percent": summary_fs["coverage_percent"],
                "CSE_count": cse_count,
                "CSE_rate_percent": pct(cse_count),
            }
            per_sample.append(item)
            csv_rows.append(item)
        metric_names = [
            "C_percent",
            "S_percent",
            "CS_percent",
            "ER_F_percent",
            "ER_FS_percent",
            "AS_F_arithmetic",
            "AS_F_geomean",
            "AS_FS_arithmetic",
            "AS_FS_geomean",
            "CSE_rate_percent",
        ]
        across_samples = {
            "num_samples": self.n_samples
        }
        for metric in metric_names:
            values = [
                item[metric]
                for item in per_sample
                if item[metric] is not None
            ]
            across_samples[f"{metric}_mean"] = _mean(values)
            across_samples[f"{metric}_std"] = _std(values)
        output = {
            "metadata": {
                "eval_path": self.eval_path,
                "n_samples": self.n_samples,
                "warmup": self.warmup,
                "repeat": self.repeat,
                "timeout_per_test": self.timeout_per_test,
                "faster_threshold": self.faster_threshold,
                "language": self.lang or "all",
                "definitions": {
                    "C": "functional correctness over all evaluated tasks",
                    "S": "security-test pass over all evaluated tasks",
                    "CS": "functional AND secure over all evaluated tasks",
                    "ER_F": (
                        "speedup > threshold among measured functional rows"
                    ),
                    "ER_FS": (
                        "speedup > threshold among measured functional+secure rows"
                    ),
                    "CSE": (
                        "functional AND secure AND speedup > threshold, "
                        "divided by all evaluated tasks"
                    ),
                },
            },
            "per_sample": per_sample,
            "across_samples": across_samples,
        }
        json_path = os.path.join(
            self.eval_path,
            "sampling_efficiency_summary.json",
        )
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        csv_path = os.path.join(
            self.eval_path,
            "sampling_efficiency_summary.csv",
        )
        if csv_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(csv_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(csv_rows)
        print("\n" + "=" * 80)
        print("Sampling C/S/E summary")
        print("=" * 80)
        def show(metric: str, label: str, suffix: str = "%"):
            mean = across_samples.get(f"{metric}_mean")
            std = across_samples.get(f"{metric}_std")

            if mean is not None:
                print(
                    f"{label:<12}: "
                    f"{mean:.4f} ± {std:.4f}{suffix}"
                )
        show("C_percent", "C")
        show("S_percent", "S")
        show("CS_percent", "CS")
        show("ER_F_percent", "ER_F")
        show("ER_FS_percent", "ER_FS")
        show("AS_F_arithmetic", "AS_F", "x")
        show("AS_FS_arithmetic", "AS_FS", "x")
        show("CSE_rate_percent", "CSE")
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
        return output
    def run(self) -> None:
        # Reuse the already-tested timing implementation.
        super().run()
        # Add per-sample and mean/std reporting for RQ3/RQ4.
        self.report_sampling_efficiency()
if __name__ == "__main__":
    fire.Fire(SamplingEfficiencyEvaler)