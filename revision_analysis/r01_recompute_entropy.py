#!/usr/bin/env python3
"""Recompute Histogram Entropy for the complete frozen five-domain corpus."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
REPOSITORY_ROOT = REVISION_ROOT.parent
sys.path.insert(0, str(HERE))
from corrected_entropy import DEFAULT_BINS, histogram_entropy

DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
EXPECTED = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
STACK_ARRAYS = REPOSITORY_ROOT / "outputs/01_stacks/arrays"
SUBMITTED = REPOSITORY_ROOT / "outputs/03_single_measure_curves/raw"
CACHE_OUT = REVISION_ROOT / "05_cached_data/corrected_entropy"
OUT = REVISION_ROOT / "06_analysis_outputs/corrected_entropy"
LOG = REVISION_ROOT / "12_logs/r01_corrected_entropy.log"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalize(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    spread = float(np.ptp(curve))
    return np.zeros_like(curve) if spread <= 1e-12 else (curve - float(curve.min())) / spread


def load_object_curves(path: Path) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.float64) for value in np.load(path, allow_pickle=True)]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    curve_rows: list[dict] = []
    peak_rows: list[dict] = []
    affected_rows: list[dict] = []
    summaries: list[dict] = []
    validation: dict = {"generated_at": now(), "domains": {}, "status": "PASS", "failures": []}

    with LOG.open("w", encoding="utf-8") as log:
        for domain in DOMAINS:
            log.write(f"{now()} loading {domain}\n"); log.flush()
            stack_path = STACK_ARRAYS / f"{domain}_stacks.npy"
            stacks = np.load(stack_path, allow_pickle=True)
            submitted = load_object_curves(SUBMITTED / domain / "histogram_entropy.npy")
            if len(stacks) != EXPECTED[domain] or len(submitted) != EXPECTED[domain]:
                validation["failures"].append(f"{domain}: stack/curve count mismatch")
                validation["status"] = "FAIL"
                continue
            corrected: list[np.ndarray] = []
            corrected_normalized: list[np.ndarray] = []
            changed_peaks = changed_values = constant_corrected = 0
            max_abs_delta = 0.0
            for stack_index, stack in enumerate(stacks):
                values = np.asarray([histogram_entropy(image, bins=DEFAULT_BINS) for image in stack], dtype=np.float64)
                norm = normalize(values)
                corrected.append(values)
                corrected_normalized.append(norm)
                old = submitted[stack_index]
                old_peak = int(np.argmax(old))
                new_peak = int(np.argmax(values))
                delta = np.abs(values - old)
                is_changed = bool(np.any(delta > 1e-12))
                changed_values += int(is_changed)
                changed_peaks += int(old_peak != new_peak)
                constant_corrected += int(np.ptp(values) <= 1e-12)
                max_abs_delta = max(max_abs_delta, float(delta.max(initial=0.0)))
                stack_id = f"{domain}_{stack_index:05d}"
                if is_changed or old_peak != new_peak:
                    affected_rows.append({"domain": domain, "stack_index": stack_index, "stack_id": stack_id, "submitted_peak": old_peak, "corrected_peak": new_peak, "peak_shift": new_peak - old_peak, "values_changed": is_changed})
                peak_rows.append({"domain": domain, "stack_index": stack_index, "stack_id": stack_id, "slice_count": len(values), "submitted_peak": old_peak, "corrected_peak": new_peak, "absolute_peak_shift": abs(new_peak - old_peak), "submitted_constant": bool(np.ptp(old) <= 1e-12), "corrected_constant": bool(np.ptp(values) <= 1e-12)})
                for slice_index, (before, after) in enumerate(zip(old, values)):
                    curve_rows.append({"domain": domain, "stack_index": stack_index, "stack_id": stack_id, "slice_index": slice_index, "submitted_entropy_bits": float(before), "corrected_entropy_bits": float(after), "absolute_delta_bits": float(abs(after - before))})
                if (stack_index + 1) % 500 == 0:
                    log.write(f"{now()} {domain} {stack_index + 1}/{len(stacks)}\n"); log.flush()

            raw_array = np.empty(len(corrected), dtype=object)
            normalized_array = np.empty(len(corrected_normalized), dtype=object)
            raw_array[:] = corrected
            normalized_array[:] = corrected_normalized
            raw_path = CACHE_OUT / f"{domain}_histogram_entropy_raw.npy"
            norm_path = CACHE_OUT / f"{domain}_histogram_entropy_normalized.npy"
            np.save(raw_path, raw_array, allow_pickle=True)
            np.save(norm_path, normalized_array, allow_pickle=True)
            summaries.append({"domain": domain, "stack_count": len(corrected), "slice_count": sum(len(x) for x in corrected), "submitted_constant_curves": sum(int(np.ptp(x) <= 1e-12) for x in submitted), "corrected_constant_curves": constant_corrected, "stacks_with_value_changes": changed_values, "stacks_with_peak_changes": changed_peaks, "max_absolute_delta_bits": max_abs_delta, "corrected_raw_cache": str(raw_path), "corrected_normalized_cache": str(norm_path)})
            validation["domains"][domain] = {"stack_count": len(corrected), "all_finite": all(np.all(np.isfinite(x)) for x in corrected), "constant_corrected_curves": constant_corrected, "raw_sha256": digest(raw_path), "normalized_sha256": digest(norm_path)}
            del stacks

    write_csv(OUT / "submitted_vs_corrected_curves.csv", curve_rows, list(curve_rows[0]))
    write_csv(OUT / "submitted_vs_corrected_peaks.csv", peak_rows, list(peak_rows[0]))
    write_csv(OUT / "submitted_vs_corrected_domain_summary.csv", summaries, list(summaries[0]))
    write_csv(OUT / "affected_stack_manifest.csv", affected_rows, list(affected_rows[0]) if affected_rows else ["domain", "stack_index", "stack_id", "submitted_peak", "corrected_peak", "peak_shift", "values_changed"])
    configuration = {"generated_at": now(), "bins": DEFAULT_BINS, "integer_range": "full representable dtype range", "float_range": [0.0, 1.0], "per_image_normalization": False, "seed": None, "input_hashes": {domain: digest(STACK_ARRAYS / f"{domain}_stacks.npy") for domain in DOMAINS}, "software": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}}
    (OUT / "run_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (OUT / "entropy_definition.md").write_text("""# Corrected Histogram Entropy definition

Each integer image is mapped from the full representable range of its stored dtype to 256 fixed equal-width bins. Thus uint8 uses [0,255] and uint16 uses [0,65535], with endpoints included. Floating images are interpreted in a declared fixed range, [0,1] by default. The method performs no per-image min-max normalization. Every finite pixel contributes to exactly one bin, probabilities are histogram counts divided by the pixel count, and entropy is reported in bits as `-sum(p * log2(p))` over non-zero bins.

This convention is invariant for equivalent uint8 content encoded as uint16 by multiplication by 257 and prevents the submitted uint16 exclusion defect.
""", encoding="utf-8")
    tbf = next(row for row in summaries if row["domain"] == "TBF")
    findings = f"""# Entropy correction findings

- The submitted implementation excluded TBF values above 256 after converting uint16 images to float.
- Submitted TBF constant curves: {tbf['submitted_constant_curves']} / {tbf['stack_count']}.
- Corrected TBF constant curves: {tbf['corrected_constant_curves']} / {tbf['stack_count']}.
- TBF stacks with a changed entropy peak: {tbf['stacks_with_peak_changes']} / {tbf['stack_count']}.
- All five domains were recomputed under the same fixed definition; see the domain summary and validation file.
- Downstream reference labels must use the corrected curves generated here.

Elapsed: {time.time() - started:.1f} seconds.
"""
    (OUT / "ENTROPY_CORRECTION_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(findings)
    return 1 if validation["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
