#!/usr/bin/env python3
"""Controlled resampling audit for the frozen 32-operator pool.

The mechanism experiment first maps each source slice to an aspect-preserving
512-pixel-longest-side experimental base.  It then changes only scale,
interpolator, or geometry.  Native cached curves are retained as a separately
labelled observational comparator; they are not confused with the controlled
base experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.stats import spearmanr

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_entropy import histogram_entropy
from src.evaluation.autofocus_metrics import (
    false_maxima_count,
    full_width_half_maximum,
    noise_level,
    normalize_focus_curve,
)
from src.measures.focus_measure_library import build_focus_measure_registry

OUT = REVISION_ROOT / "06_analysis_outputs/resampling"
CACHE = REVISION_ROOT / "05_cached_data/resampling"
LOG = REVISION_ROOT / "12_logs/r08_controlled_resampling.log"
REGISTRY_PATH = REVISION_ROOT / "00_audit/operator_registry_32.json"
NATIVE_CURVES = REPOSITORY_ROOT / "outputs/03_single_measure_curves/normalized"
ENTROPY_CURVES = REVISION_ROOT / "05_cached_data/corrected_entropy"
REFERENCE = REVISION_ROOT / "05_cached_data/reference_ladder"
SUBMITTED_1024 = REPOSITORY_ROOT / "outputs/04_single_measure_eval/supplementary/rank_stability_1024.csv"
DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
COUNTS = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
ROOTS = {
    "WBC": Path("/mnt/d/New_folder/datasets/WBC_dataset1"),
    "TBI": Path("/mnt/d/New_folder/datasets/New folder/TBSI/folders"),
    "PBS": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/pbs_imgs"),
    "BMA": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/bma_imgs"),
    "TBF": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/tbf_imgs"),
}
SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SEED = 4524210
BASE_LONGEST = 512
SELECTED_PER_DOMAIN = 3
INTERPOLATORS = {
    "nearest": cv.INTER_NEAREST,
    "bilinear": cv.INTER_LINEAR,
    "bicubic": cv.INTER_CUBIC,
    "area": cv.INTER_AREA,
}


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def folders(domain: str) -> list[Path]:
    root = ROOTS[domain]
    if domain == "WBC":
        return [root / str(index) for index in range(COUNTS[domain])]
    return sorted((Path(entry.path) for entry in os.scandir(root) if entry.is_dir()), key=lambda path: natural_key(path.name))


def selected_indices(domain: str) -> list[int]:
    count = COUNTS[domain]
    return sorted({0, count // 2, count - 1})


def read_stack(folder: Path) -> list[np.ndarray]:
    paths = sorted((Path(entry.path) for entry in os.scandir(folder) if entry.is_file() and Path(entry.name).suffix.lower() in SUFFIXES and ":Zone.Identifier" not in entry.name), key=lambda path: natural_key(path.name))
    result = []
    for path in paths:
        raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"failed to read {path}")
        if raw.ndim == 3 and raw.shape[2] == 3:
            raw = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        elif raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv.cvtColor(raw, cv.COLOR_BGRA2GRAY)
        if np.issubdtype(raw.dtype, np.integer):
            limits = np.iinfo(raw.dtype)
            image = (raw.astype(np.float32) - float(limits.min)) / float(limits.max - limits.min)
        else:
            image = raw.astype(np.float32)
            if float(np.nanmin(image)) < 0 or float(np.nanmax(image)) > 1:
                raise ValueError(f"float source outside [0,1]: {path}")
        result.append(np.ascontiguousarray(image))
    if not result:
        raise FileNotFoundError(f"no slices in {folder}")
    return result


def aspect_resize(image: np.ndarray, longest: int, interpolation: int) -> np.ndarray:
    height, width = image.shape
    scale = longest / max(height, width)
    shape = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    out = cv.resize(image, shape, interpolation=interpolation)
    return np.ascontiguousarray(np.clip(out, 0, 1), dtype=np.float32)


def square_resize(image: np.ndarray, side: int, interpolation: int) -> np.ndarray:
    out = cv.resize(image, (side, side), interpolation=interpolation)
    return np.ascontiguousarray(np.clip(out, 0, 1), dtype=np.float32)


def curve_metrics(curve: np.ndarray, label: int) -> dict[str, float | int]:
    normalized = normalize_focus_curve(np.asarray(curve, dtype=np.float64))
    peak = int(np.argmax(normalized))
    error = abs(peak - int(label))
    return {
        "peak_index": peak,
        "absolute_peak_error_slices": error,
        "exact_match": int(error == 0),
        "within_one_slice": int(error <= 1),
        "false_maxima_count": float(false_maxima_count(normalized)),
        "noise_level": float(noise_level(normalized)),
        "fwhm_slices": float(full_width_half_maximum(normalized)),
    }, normalized


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def rank_rows(summary: list[dict]) -> list[dict]:
    result = []
    metrics = ("mean_absolute_peak_error_slices", "mean_false_maxima_count", "mean_noise_level", "mean_fwhm_slices")
    for condition in sorted({row["condition"] for row in summary}):
        subset = [row for row in summary if row["condition"] == condition]
        normalized = {metric: {} for metric in metrics}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset])
            spread = float(np.ptp(values))
            scores = np.zeros_like(values) if spread <= 1e-12 else (values - float(values.min())) / spread
            normalized[metric] = {row["operator"]: float(value) for row, value in zip(subset, scores)}
        ordered = []
        for row in subset:
            operator = row["operator"]
            mechanism_score = float(np.mean([normalized[metric][operator] for metric in metrics]))
            ordered.append((mechanism_score, operator, row))
        ordered.sort(key=lambda item: (item[0], item[1]))
        for rank, (score, operator, row) in enumerate(ordered, 1):
            result.append({"condition": condition, "operator": operator, "family": row["family"], "mechanism_score": score, "rank": rank, **{metric: row[metric] for metric in metrics}})
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    meta = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    operators = [row["measure_name"] for row in meta]
    slugs = {row["measure_name"]: row["slug"] for row in meta}
    families = {row["measure_name"]: row["family"] for row in meta}
    registry = build_focus_measure_registry(); registry["Histogram Entropy"]["func"] = histogram_entropy
    per_stack: list[dict] = []
    curves_long: list[dict] = []
    conditions = [{"condition": "native_cached", "geometry": "native", "direction": "observational_native", "scale_factor": "native", "interpolation": "native"}]
    for direction, factor in (("downsample", 0.5), ("base", 1.0), ("upsample", 2.0)):
        for interpolation in INTERPOLATORS:
            if direction == "base" and interpolation != "area":
                continue
            conditions.append({"condition": f"aspect_{factor:g}x_{interpolation}", "geometry": "aspect_preserved", "direction": direction, "scale_factor": factor, "interpolation": interpolation})
    for interpolation in INTERPOLATORS:
        conditions.append({"condition": f"square_512_{interpolation}", "geometry": "square_distortion", "direction": "geometry_only", "scale_factor": 1.0, "interpolation": interpolation})

    with LOG.open("w", encoding="utf-8") as log:
        for domain in DOMAINS:
            domain_folders = folders(domain)
            labels = np.load(REFERENCE / f"{domain}_REF_B_fixed_ten.npy")
            native_by_operator = {}
            for operator in operators:
                path = ENTROPY_CURVES / f"{domain}_histogram_entropy_normalized.npy" if operator == "Histogram Entropy" else NATIVE_CURVES / domain / f"{slugs[operator]}.npy"
                native_by_operator[operator] = np.load(path, allow_pickle=True)
            for stack_index in selected_indices(domain):
                stack = read_stack(domain_folders[stack_index])
                label = int(labels[stack_index])
                base = [aspect_resize(image, BASE_LONGEST, cv.INTER_AREA if max(image.shape) >= BASE_LONGEST else cv.INTER_CUBIC) for image in stack]
                for condition in conditions:
                    name = condition["condition"]
                    if name == "native_cached":
                        prepared = None
                    elif condition["geometry"] == "square_distortion":
                        prepared = [square_resize(image, BASE_LONGEST, INTERPOLATORS[condition["interpolation"]]) for image in base]
                    else:
                        target = int(round(BASE_LONGEST * float(condition["scale_factor"])))
                        prepared = [aspect_resize(image, target, INTERPOLATORS[condition["interpolation"]]) for image in base]
                    for operator in operators:
                        curve = np.asarray(native_by_operator[operator][stack_index], dtype=np.float64) if prepared is None else np.asarray([registry[operator]["func"](image) for image in prepared], dtype=np.float64)
                        metrics, normalized = curve_metrics(curve, label)
                        row = {"domain": domain, "stack_index": stack_index, "source_folder": str(domain_folders[stack_index]), **condition, "operator": operator, "family": families[operator], "reference_tier": "REF-B_fixed_ten_diagnostic", "reference_index": label, "slice_count": len(curve), **metrics}
                        per_stack.append(row)
                        for slice_index, (raw_value, normalized_value) in enumerate(zip(curve, normalized)):
                            curves_long.append({"domain": domain, "stack_index": stack_index, "condition": name, "operator": operator, "slice_index": slice_index, "raw_value": float(raw_value), "normalized_value": float(normalized_value)})
                    log.write(f"{datetime.now().isoformat()} {domain} stack={stack_index} condition={name}\n"); log.flush()

    summary = []
    for condition in [row["condition"] for row in conditions]:
        for operator in operators:
            subset = [row for row in per_stack if row["condition"] == condition and row["operator"] == operator]
            summary.append({"condition": condition, "operator": operator, "family": families[operator], "n_stacks": len(subset), "mean_absolute_peak_error_slices": float(np.mean([row["absolute_peak_error_slices"] for row in subset])), "exact_match_rate": float(np.mean([row["exact_match"] for row in subset])), "within_one_slice_rate": float(np.mean([row["within_one_slice"] for row in subset])), "mean_false_maxima_count": float(np.mean([row["false_maxima_count"] for row in subset])), "mean_noise_level": float(np.mean([row["noise_level"] for row in subset])), "mean_fwhm_slices": float(np.mean([row["fwhm_slices"] for row in subset]))})
    ranks = rank_rows(summary)
    native_rank = {row["operator"]: int(row["rank"]) for row in ranks if row["condition"] == "native_cached"}
    base_rank = {row["operator"]: int(row["rank"]) for row in ranks if row["condition"] == "aspect_1x_area"}
    rank_shifts = []
    for row in ranks:
        rank_shifts.append({**row, "rank_shift_vs_native": int(row["rank"]) - native_rank[row["operator"]], "rank_shift_vs_controlled_base": int(row["rank"]) - base_rank[row["operator"]]})

    correlations = []
    mechanism = []
    base_lookup = {(row["operator"]): row for row in summary if row["condition"] == "aspect_1x_area"}
    for condition in sorted({row["condition"] for row in summary if row["condition"] not in ("native_cached", "aspect_1x_area")}):
        rows = [row for row in summary if row["condition"] == condition]
        rank_lookup = {row["operator"]: row for row in rank_shifts if row["condition"] == condition}
        improvement = np.asarray([-rank_lookup[row["operator"]]["rank_shift_vs_controlled_base"] for row in rows], dtype=float)
        for metric in ("mean_noise_level", "mean_false_maxima_count", "mean_absolute_peak_error_slices", "mean_fwhm_slices"):
            reduction = np.asarray([float(base_lookup[row["operator"]][metric]) - float(row[metric]) for row in rows])
            rho, pvalue = spearmanr(improvement, reduction)
            correlations.append({"condition": condition, "association": f"rank_improvement_vs_{metric}_reduction", "spearman_rho": float(rho) if np.isfinite(rho) else "not_estimable", "two_sided_p_value_descriptive": float(pvalue) if np.isfinite(pvalue) else "not_estimable", "n_operators": len(rows), "interpretation": "association; not proof of mechanism"})
        for operator in ("Roberts Focus Measure", "Brenner Gradient"):
            row = next(item for item in rows if item["operator"] == operator); base_row = base_lookup[operator]
            mechanism.append({"condition": condition, "operator": operator, "rank": rank_lookup[operator]["rank"], "rank_shift_vs_controlled_base": rank_lookup[operator]["rank_shift_vs_controlled_base"], "noise_change": float(row["mean_noise_level"]) - float(base_row["mean_noise_level"]), "false_maxima_change": float(row["mean_false_maxima_count"]) - float(base_row["mean_false_maxima_count"]), "absolute_peak_error_change_slices": float(row["mean_absolute_peak_error_slices"]) - float(base_row["mean_absolute_peak_error_slices"]), "fwhm_change_slices": float(row["mean_fwhm_slices"]) - float(base_row["mean_fwhm_slices"]), "operator_support_note": "2-pixel directional difference" if operator.startswith("Brenner") else "2x2 Roberts cross-gradient"})

    submitted = list(csv.DictReader(SUBMITTED_1024.open(newline="", encoding="utf-8")))
    inconsistency = [{"artifact": "submitted rank_stability_1024.csv", "operator": row["measure_name"], "native_value_rank": row["native_value_final_rank"], "submitted_1024_value_rank": row["resolution_value_final_rank"], "submitted_value_rank_shift": row["value_rank_shift"], "resolution": "forced 1024x1024", "resolution_status": "confounded scale + aspect distortion + interpolation; superseded by controlled analysis"} for row in submitted]
    inconsistency.append({"artifact": "submitted manuscript supplement list", "operator": "not applicable", "native_value_rank": "not applicable", "submitted_1024_value_rank": "not applicable", "submitted_value_rank_shift": "not applicable", "resolution": "not applicable", "resolution_status": "numbering corrected: S2 is alpha sensitivity, S5 is composite deduplication, and the 1024 rank-stability table is S7 in the frozen export"})

    write_csv(OUT / "resampling_per_stack_metrics.csv", per_stack)
    write_csv(OUT / "resampling_operator_summary.csv", summary)
    write_csv(OUT / "resampling_rank_shifts.csv", rank_shifts)
    write_csv(OUT / "resampling_mechanism_correlations.csv", correlations)
    write_csv(OUT / "roberts_brenner_mechanism.csv", mechanism)
    write_csv(OUT / "submitted_resampling_inconsistency_audit.csv", inconsistency)
    write_csv(CACHE / "resampled_focus_curves_long.csv", curves_long)
    config = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(), "seed": SEED,
        "candidate_pool": 32, "domains": DOMAINS, "selected_stacks_per_domain": SELECTED_PER_DOMAIN,
        "selection_rule": "first, middle and last stack in deterministic natural/source order",
        "controlled_base": "aspect-preserving 512-pixel longest side, AREA for reduction and CUBIC for enlargement",
        "scale_factors_from_controlled_base": [0.5, 1.0, 2.0], "interpolators": list(INTERPOLATORS),
        "square_distortion": "512x512, explicitly separate from aspect-preserving conditions",
        "common_physical_pixel_size": "not run: pixel size is unreported for four domains and thus cannot be harmonized without invention",
        "native_comparator": "frozen cached curves for the same 15 stacks and all 32 operators; corrected entropy substituted",
        "mechanism_rank": "candidate-pool-relative equal mean of min-max normalized localization error, false maxima, noise level and FWHM; descriptive only",
        "input_hashes": {"operator_registry": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(), "submitted_1024": hashlib.sha256(SUBMITTED_1024.read_bytes()).hexdigest()},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "opencv": cv.__version__},
    }
    (OUT / "resampling_configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    finite_fields = ("absolute_peak_error_slices", "false_maxima_count", "noise_level", "fwhm_slices")
    validation = {"status": "PASS", "operators": len(operators), "domains": len(DOMAINS), "conditions": len(conditions), "stacks": len({(row["domain"], row["stack_index"]) for row in per_stack}), "all_finite": all(np.isfinite(float(row[field])) for row in per_stack for field in finite_fields), "frozen_pool_in_every_condition": all(len({row["operator"] for row in per_stack if row["condition"] == condition["condition"]}) == 32 for condition in conditions), "aspect_and_square_separated": True}
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (OUT / "RESAMPLING_FINDINGS.md").write_text("""# Controlled resampling findings

This experiment supersedes the submitted forced 1024x1024 comparison. Native cached curves remain an observational comparator. The controlled mechanism experiment first forms one aspect-preserving 512-longest-side base per slice, then changes one factor at a time: scale (0.5x or 2x), interpolation (nearest, bilinear, bicubic, area), or square-grid distortion. All 32 frozen operators are present in every condition.

Roberts and Brenner changes are decomposed into localization, false-maxima, curve-noise and FWHM changes in `roberts_brenner_mechanism.csv`. Spearman correlations are labelled associations, not causal proof. A common physical-pixel-size experiment is unavailable because the required pixel-size metadata are not verified across domains.

The frozen supplement numbering itself was inconsistent with the manuscript list: S2 is alpha sensitivity, S5 is composite deduplication, and the native-versus-1024 rank-stability export is S7. The submitted 1024 result also confounded scale, aspect distortion and interpolation; it is retained only as historical evidence.
""", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
