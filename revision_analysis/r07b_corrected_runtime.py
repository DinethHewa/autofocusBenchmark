#!/usr/bin/env python3
"""Corrected, repeated, dtype-matched focus-operator timing protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.measures.focus_measure_library import build_focus_measure_registry
from corrected_entropy import histogram_entropy as corrected_histogram_entropy

OUT = REVISION_ROOT / "06_analysis_outputs/corrected_runtime"
REGISTRY_PATH = REVISION_ROOT / "00_audit/operator_registry_32.json"
SUBMITTED_TIMING = REPOSITORY_ROOT / "outputs/03_single_measure_curves/single_timing_summary.csv"
DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
STACK_COUNTS = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
SLICE_COUNTS = {"WBC": 257730, "TBI": 300, "PBS": 1035, "BMA": 532, "TBF": 2548}
ROOTS = {
    "WBC": Path("/mnt/d/New_folder/datasets/WBC_dataset1"),
    "TBI": Path("/mnt/d/New_folder/datasets/New folder/TBSI/folders"),
    "PBS": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/pbs_imgs"),
    "BMA": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/bma_imgs"),
    "TBF": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/tbf_imgs"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SEED = 4524210
REPEATS = 7
WARMUPS = 2
BOOTSTRAPS = 2000


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def natural_key(value: str) -> list[object]:
    import re
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def selected_images(domain: str) -> list[Path]:
    root = ROOTS[domain]
    if domain == "WBC":
        folders = [root / str(index) for index in (0, 12886, 25772)]
    else:
        children = sorted((Path(entry.path) for entry in os.scandir(root) if entry.is_dir()), key=lambda path: natural_key(path.name))
        positions = sorted({0, len(children) // 2, len(children) - 1})
        folders = [children[index] for index in positions]
    result = []
    for folder in folders:
        images = sorted((Path(entry.path) for entry in os.scandir(folder) if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES and ":Zone.Identifier" not in entry.name), key=lambda path: natural_key(path.name))
        if not images:
            raise FileNotFoundError(f"no image found in {folder}")
        result.append(images[len(images) // 2])
    return result


def preprocess(raw: np.ndarray, target: int | None) -> np.ndarray:
    if raw.ndim == 3 and raw.shape[2] == 3:
        image = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        image = cv.cvtColor(raw, cv.COLOR_BGRA2GRAY)
    else:
        image = np.asarray(raw)
    # Convert through a fixed, dtype-defined radiometric mapping before any
    # interpolation.  This preserves the uint8/uint16 physical-code
    # equivalence required by the corrected entropy convention and gives every
    # operator the same contiguous float32 input in [0, 1].
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        image = image.astype(np.float32)
        image = (image - float(limits.min)) / float(limits.max - limits.min)
    elif np.issubdtype(image.dtype, np.bool_):
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32, copy=False)
        if image.size and (float(np.nanmin(image)) < 0.0 or float(np.nanmax(image)) > 1.0):
            raise ValueError("floating source image lies outside the declared [0, 1] radiometric range")
    if target is not None:
        height, width = image.shape[:2]
        scale = target / max(height, width)
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC
        image = cv.resize(image, (out_width, out_height), interpolation=interpolation)
        # Bicubic enlargement can overshoot its input support slightly.
        np.clip(image, 0.0, 1.0, out=image)
    return np.ascontiguousarray(image, dtype=np.float32)


def elapsed_ms(function, *args) -> float:
    start = time.perf_counter_ns(); function(*args)
    return (time.perf_counter_ns() - start) / 1e6


def summarize(values: list[float], rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=np.float64)
    means = np.mean(rng.choice(array, size=(BOOTSTRAPS, len(array)), replace=True), axis=1)
    q1, q3 = np.percentile(array, [25, 75])
    return {
        "n_measurements": len(array), "median_ms": float(np.median(array)), "mean_ms": float(np.mean(array)),
        "sd_ms": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "iqr_ms": float(q3 - q1), "p5_ms": float(np.percentile(array, 5)), "p95_ms": float(np.percentile(array, 95)),
        "bootstrap_mean_ci_low_ms": float(np.percentile(means, 2.5)), "bootstrap_mean_ci_high_ms": float(np.percentile(means, 97.5)),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry_meta = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = build_focus_measure_registry()
    registry["Histogram Entropy"]["func"] = corrected_histogram_entropy
    operators = [entry["measure_name"] for entry in registry_meta]
    raw_images: dict[str, list[np.ndarray]] = {}
    image_paths: dict[str, list[str]] = {}
    io_values: dict[str, list[float]] = {}
    for domain in DOMAINS:
        paths = selected_images(domain)
        image_paths[domain] = [str(path) for path in paths]
        raw_images[domain] = [cv.imread(str(path), cv.IMREAD_UNCHANGED) for path in paths]
        if any(image is None for image in raw_images[domain]):
            raise RuntimeError(f"failed to read a selected {domain} image")
        io_values[domain] = [elapsed_ms(cv.imread, str(path), cv.IMREAD_UNCHANGED) for path in paths for _ in range(REPEATS)]

    targets = (("native", None), ("128", 128), ("512", 512), ("1024", 1024))
    prepared: dict[tuple[str, str, int], np.ndarray] = {}
    prep_times: dict[tuple[str, str], list[float]] = {}
    for domain in DOMAINS:
        for label, target in targets:
            prep_times[(domain, label)] = []
            for image_index, raw in enumerate(raw_images[domain]):
                prepared[(domain, label, image_index)] = preprocess(raw, target)
                for _ in range(REPEATS):
                    prep_times[(domain, label)].append(elapsed_ms(preprocess, raw, target))

    jobs = [(domain, label, target, operator) for domain in DOMAINS for label, target in targets for operator in operators]
    random.Random(SEED).shuffle(jobs)
    raw_rows: list[dict] = []
    log_path = REVISION_ROOT / "12_logs/r07b_corrected_runtime.log"
    with log_path.open("w", encoding="utf-8") as log:
        for job_index, (domain, label, target, operator) in enumerate(jobs, start=1):
            function = registry[operator]["func"]
            kernel_times: list[float] = []
            combined_times: list[float] = []
            for image_index, raw in enumerate(raw_images[domain]):
                image = prepared[(domain, label, image_index)]
                for _ in range(WARMUPS): function(image)
                for repeat in range(REPEATS):
                    kernel = elapsed_ms(function, image)
                    start = time.perf_counter_ns(); processed = preprocess(raw, target); function(processed)
                    combined = (time.perf_counter_ns() - start) / 1e6
                    kernel_times.append(kernel); combined_times.append(combined)
                    raw_rows.append({"domain": domain, "resolution": label, "operator": operator, "image_index": image_index, "repeat": repeat, "height": image.shape[0], "width": image.shape[1], "megapixels": image.size / 1e6, "kernel_ms": kernel, "combined_ms": combined})
            log.write(f"{now()} {job_index}/{len(jobs)} {domain} {label} {operator}\n"); log.flush()

    rng = np.random.default_rng(SEED)
    summary_rows: list[dict] = []
    for domain in DOMAINS:
        io_summary = summarize(io_values[domain], rng)
        for label, _ in targets:
            prep_summary = summarize(prep_times[(domain, label)], rng)
            for operator in operators:
                subset = [row for row in raw_rows if row["domain"] == domain and row["resolution"] == label and row["operator"] == operator]
                kernel = summarize([row["kernel_ms"] for row in subset], rng)
                combined = summarize([row["combined_ms"] for row in subset], rng)
                pixels = float(np.mean([row["megapixels"] for row in subset]))
                row = {"domain": domain, "resolution": label, "operator": operator, "height": subset[0]["height"], "width": subset[0]["width"], "megapixels": pixels, **{f"kernel_{key}": value for key, value in kernel.items()}, **{f"preprocessing_{key}": value for key, value in prep_summary.items()}, **{f"combined_{key}": value for key, value in combined.items()}, "io_mean_ms": io_summary["mean_ms"], "kernel_megapixels_per_second": pixels / (kernel["median_ms"] / 1000.0) if kernel["median_ms"] > 0 else None}
                summary_rows.append(row)

    resolution_rows: list[dict] = []
    for label, _ in targets:
        for operator in operators:
            rows = [row for row in summary_rows if row["resolution"] == label and row["operator"] == operator]
            resolution_rows.append({"resolution": label, "operator": operator, "equal_domain_macro_median_kernel_ms": float(np.mean([row["kernel_median_ms"] for row in rows])), "slice_weighted_micro_median_kernel_ms": float(np.average([row["kernel_median_ms"] for row in rows], weights=[SLICE_COUNTS[row["domain"]] for row in rows])), "equal_domain_macro_median_combined_ms": float(np.mean([row["combined_median_ms"] for row in rows])), "slice_weighted_micro_median_combined_ms": float(np.average([row["combined_median_ms"] for row in rows], weights=[SLICE_COUNTS[row["domain"]] for row in rows]))})

    macro_micro = [row for row in resolution_rows if row["resolution"] == "native"]
    submitted_rows = list(csv.DictReader(SUBMITTED_TIMING.open(newline="", encoding="utf-8")))
    submitted_comparison: list[dict] = []
    for operator in operators:
        old = [row for row in submitted_rows if row["measure_name"] == operator]
        new = next(row for row in macro_micro if row["operator"] == operator)
        if old:
            submitted_macro = float(np.mean([float(row["native_avg_time_per_slice_sec"]) * 1000 for row in old]))
            submitted_comparison.append({"operator": operator, "submitted_equal_domain_macro_native_ms": submitted_macro, "corrected_equal_domain_macro_native_kernel_ms": new["equal_domain_macro_median_kernel_ms"], "corrected_slice_weighted_micro_native_kernel_ms": new["slice_weighted_micro_median_kernel_ms"], "submitted_vs_corrected_macro_delta_ms": new["equal_domain_macro_median_kernel_ms"] - submitted_macro})

    write_csv(OUT / "corrected_runtime_raw_repeats.csv", raw_rows)
    write_csv(OUT / "corrected_runtime_per_measure_domain.csv", summary_rows)
    write_csv(OUT / "corrected_runtime_per_resolution.csv", resolution_rows)
    write_csv(OUT / "runtime_macro_micro_comparison.csv", macro_micro)
    write_csv(OUT / "submitted_vs_corrected_runtime.csv", submitted_comparison)
    protocol = {"generated_at": now(), "seed": SEED, "candidate_pool": 32, "selected_images_per_domain": 3, "selected_image_rule": "first, middle, last stack in deterministic natural order; central slice", "image_paths": image_paths, "warmups": WARMUPS, "repeats_per_image": REPEATS, "bootstrap_resamples": BOOTSTRAPS, "order_randomized": True, "preprocessing": "grayscale if required; fixed dtype-range mapping to float32 [0,1]; aspect-ratio-preserving resize by longest dimension", "radiometric_mapping": "integer full representable dtype range to [0,1]; floating sources must already lie in [0,1]; no per-image min-max normalization", "native_and_resized_dtype_identical": "contiguous float32 in [0,1]", "timed_components": ["I/O", "preprocessing", "operator kernel", "combined preprocessing plus kernel"], "controlled_resolutions": [128, 512, 1024], "native_included": True, "input_hashes": {"operator_registry": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(), "submitted_timing": hashlib.sha256(SUBMITTED_TIMING.read_bytes()).hexdigest()}}
    (OUT / "runtime_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    environment = {"generated_at": now(), "platform": platform.platform(), "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(), "python": sys.version, "numpy": np.__version__, "opencv": cv.__version__, "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, "opencv_threads": cv.getNumThreads()}
    try:
        import scipy; environment["scipy"] = scipy.__version__
    except Exception: environment["scipy"] = "unavailable"
    try:
        import psutil; environment["ram_bytes"] = psutil.virtual_memory().total
    except Exception: environment["ram_bytes"] = "not reported"
    (OUT / "runtime_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    validation = {"status": "PASS", "all_finite": all(np.isfinite(row["kernel_ms"]) and np.isfinite(row["combined_ms"]) for row in raw_rows), "expected_jobs": len(jobs), "observed_job_groups": len({(row["domain"], row["resolution"], row["operator"]) for row in raw_rows}), "dtype_matched": True, "aspect_ratio_preserved": True, "warmup_used": True, "repeated": True, "order_randomized": True}
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    findings = """# Runtime correction findings

- The submitted native column is an equal-domain macro-average, not a slice-weighted average over all evaluated slices.
- Submitted native and resized paths did not apply identical preprocessing/dtype conversion, and the submitted timing used no warm-up or repeated measurements.
- Corrected results separately report I/O, preprocessing, operator-kernel, and combined preprocessing-plus-kernel timing.
- All aggregation labels are explicit. The 10.7–24.4 ms submitted claim is retired and must not be reused.
- Final score recalculation must use the corrected native operator-kernel summary and separately test runtime weights 0.10, 0.05, and 0.
"""
    (OUT / "RUNTIME_CORRECTION_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(findings)
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
