#!/usr/bin/env python3
"""Recompute the entropy RRMSE criterion under the submitted noise protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_entropy import histogram_entropy

DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
STACKS = REPOSITORY_ROOT / "outputs/01_stacks/arrays"
CORRECTED = REVISION_ROOT / "05_cached_data/corrected_entropy"
OUT = REVISION_ROOT / "06_analysis_outputs/corrected_entropy"
SEED = 123
NOISE_STD = 0.01
CAP = 200


def normalize(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    spread = float(np.ptp(curve))
    return np.zeros_like(curve) if spread <= 1e-12 else (curve - float(curve.min())) / spread


def noisy_unit_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    unit = np.zeros_like(array, dtype=np.float32) if high - low <= 1e-12 else (array - low) / (high - low)
    unit += rng.normal(0.0, NOISE_STD, size=array.shape).astype(np.float32)
    return np.clip(unit, 0.0, 1.0)


def main() -> int:
    rows = []
    log_path = REVISION_ROOT / "12_logs/r01b_entropy_noise_robustness.log"
    with log_path.open("w", encoding="utf-8") as log:
        for domain in DOMAINS:
            stacks = np.load(STACKS / f"{domain}_stacks.npy", allow_pickle=True)
            curves = [np.asarray(value, dtype=np.float64) for value in np.load(CORRECTED / f"{domain}_histogram_entropy_normalized.npy", allow_pickle=True)]
            rng = np.random.default_rng(SEED)
            values = []
            limit = min(CAP, len(stacks))
            for index in range(limit):
                noisy_curve = np.asarray([histogram_entropy(noisy_unit_image(image, rng)) for image in stacks[index]], dtype=np.float64)
                noisy_curve = normalize(noisy_curve)
                clean = curves[index]
                numerator = float(np.sqrt(np.mean((clean - noisy_curve) ** 2)))
                denominator = float(np.sqrt(np.mean(clean ** 2)) + 1e-12)
                rrmse = numerator / denominator
                values.append(rrmse)
                rows.append({"domain": domain, "stack_index": index, "rrmse": rrmse})
                if (index + 1) % 25 == 0:
                    log.write(f"{datetime.now().isoformat()} {domain} {index + 1}/{limit}\n"); log.flush()
            del stacks
    with (OUT / "corrected_entropy_rrmse_per_stack.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "stack_index", "rrmse"]); writer.writeheader(); writer.writerows(rows)
    summary = []
    for domain in DOMAINS:
        values = np.asarray([row["rrmse"] for row in rows if row["domain"] == domain])
        summary.append({"domain": domain, "n_stacks": len(values), "mean_rrmse": float(np.mean(values)), "median_rrmse": float(np.median(values)), "sd_rrmse": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})
    with (OUT / "corrected_entropy_rrmse_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    configuration = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(), "seed": SEED, "noise_std_unit_intensity": NOISE_STD, "stack_cap_per_domain": CAP, "same_seed_reset_per_domain": True, "input_hashes": {domain: hashlib.sha256((CORRECTED / f"{domain}_histogram_entropy_normalized.npy").read_bytes()).hexdigest() for domain in DOMAINS}, "validation": {"all_finite": all(np.isfinite(row["rrmse"]) for row in rows), "domains": DOMAINS}}
    (OUT / "rrmse_configuration_and_validation.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
