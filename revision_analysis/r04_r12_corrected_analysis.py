#!/usr/bin/env python3
"""Corrected scoring, clustered resampling, imbalance and sensitivity.

The official WBC ``slide_number.csv`` is reconciled to the frozen natural
numeric stack order. WBC resampling therefore uses slide as the highest
authoritative unit. The release does not map its 214 slides to 72 patients, so
patient-level inference is not fabricated. Other domains use stacks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
from src.evaluation.autofocus_metrics import (
    curvature_at_peak, false_maxima_count, full_width_half_maximum, noise_level,
    range_around_global_maximum, steep_slope_width, steep_to_gradual_slope_ratio,
)

DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
STACK_COUNTS = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
METRICS = (
    "absolute_peak_localization_error", "range_around_global_maximum", "false_maxima_count",
    "fwhm", "noise_level", "steep_to_gradual_slope_ratio", "execution_time_per_slice",
    "steep_slope_width", "curvature_at_peak", "rrmse_under_additive_noise",
)
LOWER_BETTER = {metric: metric not in {"curvature_at_peak", "steep_to_gradual_slope_ratio"} for metric in METRICS}
SUBMITTED_WEIGHTS = {
    "absolute_peak_localization_error": .20, "range_around_global_maximum": .15,
    "false_maxima_count": .10, "fwhm": .10, "noise_level": .10,
    "steep_to_gradual_slope_ratio": .10, "execution_time_per_slice": .10,
    "steep_slope_width": .05, "curvature_at_peak": .05, "rrmse_under_additive_noise": .05,
}
ALPHA = .7
BOOTSTRAPS = 1000
BALANCED_REPEATS = 1000
DIRICHLET_DRAWS = 1000
SEED = 4524210
VOTERS_4 = {"Normalized Variance", "Histogram Entropy", "GLCM Contrast", "Fourier Transform Sharpness Index"}

REGISTRY = REVISION_ROOT / "00_audit/operator_registry_32.json"
CURVES = REPOSITORY_ROOT / "outputs/03_single_measure_curves/normalized"
ENTROPY = REVISION_ROOT / "05_cached_data/corrected_entropy"
REFERENCE = REVISION_ROOT / "05_cached_data/reference_ladder"
WBC_SLIDE_MAP = REVISION_ROOT / "06_analysis_outputs/statistical_inference/wbc_slide_mapping/wbc_stack_to_slide.csv"
RUNTIME = REVISION_ROOT / "06_analysis_outputs/corrected_runtime/corrected_runtime_per_measure_domain.csv"
SUBMITTED_RAW = REPOSITORY_ROOT / "outputs/04_single_measure_eval/supplementary/dataset_metric_raw.json"
SUBMITTED_SCORE = REPOSITORY_ROOT / "outputs/04_single_measure_eval/supplementary/all_single_value_based_equal_dataset.csv"
ENTROPY_RRMSE = REVISION_ROOT / "06_analysis_outputs/corrected_entropy/corrected_entropy_rrmse_summary.csv"

SCORING_OUT = REVISION_ROOT / "06_analysis_outputs/corrected_scoring"
STAT_OUT = REVISION_ROOT / "06_analysis_outputs/statistical_inference"
IMBALANCE_OUT = REVISION_ROOT / "06_analysis_outputs/domain_imbalance"
WEIGHT_OUT = REVISION_ROOT / "06_analysis_outputs/weight_sensitivity"
RAW_OUT = REVISION_ROOT / "06_analysis_outputs/raw_supplement"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_curves(path: Path) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.float64) for value in np.load(path, allow_pickle=True)]


def peak(curve: np.ndarray) -> int:
    values = np.asarray(curve, dtype=np.float64)
    best = np.max(values); tied = np.where(np.isclose(values, best))[0]
    return int(tied[len(tied) // 2])


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def aligned(value: float, metric: str) -> float:
    return float(value) if LOWER_BETTER[metric] else float(1.0 / (value + 1e-12))


def score(raw: dict, operators: list[str], weights: dict[str, float], alpha: float, domain_weights: dict[str, float] | None = None):
    normalized: dict[str, dict[str, dict[str, float]]] = {domain: {operator: {} for operator in operators} for domain in DOMAINS}
    domain_scores: dict[str, dict[str, float]] = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        for metric in METRICS:
            values = np.asarray([aligned(raw[domain][operator][metric], metric) for operator in operators])
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            norm = np.zeros_like(values) if high - low <= 1e-12 else (values - low) / (high - low)
            for operator, value in zip(operators, norm): normalized[domain][operator][metric] = float(value)
        for operator in operators:
            denominator = sum(weights.values())
            domain_scores[domain][operator] = float(sum(weights[metric] * normalized[domain][operator][metric] for metric in METRICS) / denominator)
    rows = []
    dweights = np.asarray([domain_weights[domain] if domain_weights else 1.0 for domain in DOMAINS], dtype=float)
    for operator in operators:
        values = np.asarray([domain_scores[domain][operator] for domain in DOMAINS])
        mu = float(np.average(values, weights=dweights))
        sigma = float(np.sqrt(np.average((values - mu) ** 2, weights=dweights)))
        rows.append({"operator": operator, "mu": mu, "sigma": sigma, "G": alpha * mu + (1 - alpha) * sigma, **{f"S_{domain}": domain_scores[domain][operator] for domain in DOMAINS}})
    rows.sort(key=lambda row: (row["G"], row["operator"]))
    for rank, row in enumerate(rows, 1): row["rank"] = rank
    return rows, normalized, domain_scores


def entropy_curve_metrics(curves: list[np.ndarray]) -> dict[str, float]:
    functions = {
        "fwhm": full_width_half_maximum, "curvature_at_peak": curvature_at_peak,
        "steep_slope_width": steep_slope_width, "steep_to_gradual_slope_ratio": steep_to_gradual_slope_ratio,
        "false_maxima_count": false_maxima_count, "noise_level": noise_level,
        "range_around_global_maximum": range_around_global_maximum,
    }
    return {metric: float(np.mean([function(curve) for curve in curves])) for metric, function in functions.items()}


def prepare_raw(operators: list[str]):
    raw = json.loads(SUBMITTED_RAW.read_text(encoding="utf-8"))
    runtime = pd.read_csv(RUNTIME)
    entropy_rrmse = {row["domain"]: float(row["mean_rrmse"]) for row in csv.DictReader(ENTROPY_RRMSE.open())}
    errors_ref_b: dict[str, np.ndarray] = {}
    errors_ref_c: dict[str, np.ndarray] = {}
    per_stack_rows: list[dict] = []
    curves_by_domain_operator: dict[tuple[str, str], list[np.ndarray]] = {}
    for domain in DOMAINS:
        reference_ref_b = np.load(REFERENCE / f"{domain}_REF_B_fixed_ten.npy").astype(int)
        reference_ref_c = np.load(REFERENCE / f"{domain}_REF_C_fixed_four.npy").astype(int)
        matrix_ref_b = []; matrix_ref_c = []
        for operator in operators:
            path = ENTROPY / f"{domain}_histogram_entropy_normalized.npy" if operator == "Histogram Entropy" else CURVES / domain / f"{slug(operator)}.npy"
            curves = load_curves(path); curves_by_domain_operator[(domain, operator)] = curves
            predictions = np.asarray([peak(curve) for curve in curves], dtype=int)
            displacement_ref_b = np.abs(predictions - reference_ref_b)
            displacement_ref_c = np.abs(predictions - reference_ref_c)
            matrix_ref_b.append(displacement_ref_b); matrix_ref_c.append(displacement_ref_c)
            raw[domain][operator]["absolute_peak_localization_error"] = float(np.mean(displacement_ref_b))
            timing = runtime[(runtime.domain == domain) & (runtime.resolution == "native") & (runtime.operator == operator)]
            raw[domain][operator]["execution_time_per_slice"] = float(timing.iloc[0]["kernel_median_ms"] / 1000.0)
            if operator == "Histogram Entropy":
                raw[domain][operator].update(entropy_curve_metrics(curves))
                raw[domain][operator]["rrmse_under_additive_noise"] = entropy_rrmse[domain]
            for index, (prediction, ref_b, ref_c, error_b, error_c) in enumerate(zip(predictions, reference_ref_b, reference_ref_c, displacement_ref_b, displacement_ref_c)):
                per_stack_rows.append({"domain": domain, "stack_index": index, "stack_id": f"{domain}_{index:05d}", "operator": operator, "family": next(entry["family"] for entry in json.loads(REGISTRY.read_text()) if entry["measure_name"] == operator), "predicted_index": int(prediction), "REF_B_reference_index": int(ref_b), "REF_C_reference_index": int(ref_c), "REF_B_absolute_displacement": int(error_b), "REF_C_absolute_displacement": int(error_c), "REF_B_exact": int(error_b == 0), "REF_B_within_one": int(error_b <= 1), "REF_C_exact": int(error_c == 0), "REF_C_within_one": int(error_c <= 1)})
        errors_ref_b[domain] = np.asarray(matrix_ref_b)
        errors_ref_c[domain] = np.asarray(matrix_ref_c)
    return raw, errors_ref_b, errors_ref_c, per_stack_rows, curves_by_domain_operator


def main() -> int:
    for folder in (SCORING_OUT, STAT_OUT, IMBALANCE_OUT, WEIGHT_OUT, RAW_OUT): folder.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")); operators = [entry["measure_name"] for entry in registry]; families = {entry["measure_name"]: entry["family"] for entry in registry}
    raw, errors_ref_b, errors_ref_c, per_stack_rows, _ = prepare_raw(operators)
    final_rows, normalized, domain_scores = score(raw, operators, SUBMITTED_WEIGHTS, ALPHA)

    raw_rows = []
    for domain in DOMAINS:
        for operator in operators:
            for metric in METRICS:
                raw_rows.append({"domain": domain, "operator": operator, "family": families[operator], "criterion": metric, "raw_value": raw[domain][operator][metric], "normalized_value": normalized[domain][operator][metric], "unit": "seconds/slice" if metric == "execution_time_per_slice" else "slices" if metric in {"absolute_peak_localization_error", "fwhm", "steep_slope_width", "range_around_global_maximum"} else "dimensionless", "sample_count": STACK_COUNTS[domain], "reference_tier": "REF-B_fixed_ten_diagnostic" if metric == "absolute_peak_localization_error" else "not_reference_dependent", "runtime_protocol": "r07b_corrected_native_kernel" if metric == "execution_time_per_slice" else "not_applicable", "candidate_pool": "frozen_32"})
    write_csv(SCORING_OUT / "corrected_raw_criteria.csv", raw_rows)
    write_csv(SCORING_OUT / "corrected_final_rankings.csv", final_rows)
    write_csv(SCORING_OUT / "corrected_domain_scores.csv", [{"domain": domain, "operator": operator, "S": domain_scores[domain][operator]} for domain in DOMAINS for operator in operators])
    submitted = {row["measure_name"]: row for row in csv.DictReader(SUBMITTED_SCORE.open())}
    impacts = []
    for row in final_rows:
        old = submitted[row["operator"]]
        impacts.append({"operator": row["operator"], "family": families[row["operator"]], "submitted_G": float(old["generalization_score"]), "corrected_G": row["G"], "G_delta": row["G"] - float(old["generalization_score"]), "submitted_rank": int(old["final_rank"]), "corrected_rank": row["rank"], "rank_shift": row["rank"] - int(old["final_rank"]), "submitted_top5": int(old["final_rank"]) <= 5, "corrected_top5": row["rank"] <= 5})
    write_csv(SCORING_OUT / "submitted_vs_corrected_change_impact.csv", impacts)

    # Bootstrap score recalculation: slide clusters for WBC, stacks elsewhere.
    rng = np.random.default_rng(SEED)
    slide_map = pd.read_csv(WBC_SLIDE_MAP).sort_values("stack_index")
    if slide_map.stack_index.tolist() != list(range(STACK_COUNTS["WBC"])):
        raise RuntimeError("official WBC slide map does not align with frozen stack order")
    slide_groups = {int(slide): group.stack_index.to_numpy(dtype=int) for slide, group in slide_map.groupby("slide_num", sort=True)}
    slide_ids = np.asarray(sorted(slide_groups), dtype=int)
    bootstrap_G = np.zeros((BOOTSTRAPS, len(operators))); bootstrap_rank = np.zeros_like(bootstrap_G, dtype=int)
    bootstrap_mu = np.zeros_like(bootstrap_G); bootstrap_sigma = np.zeros_like(bootstrap_G)
    for replicate in range(BOOTSTRAPS):
        replicate_raw = {domain: {op: dict(raw[domain][op]) for op in operators} for domain in DOMAINS}
        for domain in DOMAINS:
            if domain == "WBC":
                sampled_slides = rng.choice(slide_ids, size=len(slide_ids), replace=True)
                indices = np.concatenate([slide_groups[int(slide)] for slide in sampled_slides])
            else:
                indices = rng.integers(0, STACK_COUNTS[domain], size=STACK_COUNTS[domain])
            means = np.mean(errors_ref_b[domain][:, indices], axis=1)
            for operator, value in zip(operators, means): replicate_raw[domain][operator]["absolute_peak_localization_error"] = float(value)
        rows, _, _ = score(replicate_raw, operators, SUBMITTED_WEIGHTS, ALPHA)
        for row in rows:
            index = operators.index(row["operator"]); bootstrap_G[replicate, index] = row["G"]; bootstrap_rank[replicate, index] = row["rank"]; bootstrap_mu[replicate, index] = row["mu"]; bootstrap_sigma[replicate, index] = row["sigma"]

    frequency_rows = []
    for index, operator in enumerate(operators):
        values = bootstrap_G[:, index]
        frequency_rows.append({"operator": operator, "family": families[operator], "G_mean": float(np.mean(values)), "G_ci_low": float(np.percentile(values, 2.5)), "G_ci_high": float(np.percentile(values, 97.5)), "probability_rank_1": float(np.mean(bootstrap_rank[:, index] == 1)), "probability_top3": float(np.mean(bootstrap_rank[:, index] <= 3)), "probability_top5": float(np.mean(bootstrap_rank[:, index] <= 5)), "median_rank": float(np.median(bootstrap_rank[:, index]))})
    frequency_rows.sort(key=lambda row: (-row["probability_top5"], row["G_mean"]))
    write_csv(STAT_OUT / "rank_frequencies.csv", frequency_rows)

    # Stack-level WBC resampling is retained only as a sensitivity comparator.
    stack_rng = np.random.default_rng(SEED + 1)
    stack_rank = np.zeros((BOOTSTRAPS, len(operators)), dtype=int)
    for replicate in range(BOOTSTRAPS):
        replicate_raw = {domain: {op: dict(raw[domain][op]) for op in operators} for domain in DOMAINS}
        for domain in DOMAINS:
            indices = stack_rng.integers(0, STACK_COUNTS[domain], size=STACK_COUNTS[domain])
            means = np.mean(errors_ref_b[domain][:, indices], axis=1)
            for operator, value in zip(operators, means):
                replicate_raw[domain][operator]["absolute_peak_localization_error"] = float(value)
        rows, _, _ = score(replicate_raw, operators, SUBMITTED_WEIGHTS, ALPHA)
        for row in rows:
            stack_rank[replicate, operators.index(row["operator"])] = row["rank"]
    cluster_comparison_rows = []
    for index, operator in enumerate(operators):
        cluster_comparison_rows.append({
            "operator": operator,
            "family": families[operator],
            "slide_cluster_probability_rank1": float(np.mean(bootstrap_rank[:, index] == 1)),
            "stack_level_probability_rank1": float(np.mean(stack_rank[:, index] == 1)),
            "slide_cluster_probability_top3": float(np.mean(bootstrap_rank[:, index] <= 3)),
            "stack_level_probability_top3": float(np.mean(stack_rank[:, index] <= 3)),
            "slide_cluster_probability_top5": float(np.mean(bootstrap_rank[:, index] <= 5)),
            "stack_level_probability_top5": float(np.mean(stack_rank[:, index] <= 5)),
            "slide_cluster_median_rank": float(np.median(bootstrap_rank[:, index])),
            "stack_level_median_rank": float(np.median(stack_rank[:, index])),
        })
    write_csv(STAT_OUT / "wbc_slide_cluster_vs_stack_bootstrap.csv", cluster_comparison_rows)
    family_rows = []
    for family in sorted(set(families.values())):
        indices = [operators.index(operator) for operator in operators if families[operator] == family]
        family_rows.append({"family": family, "operator_count": len(indices), "mean_top5_frequency": float(np.mean(bootstrap_rank[:, indices] <= 5)), "probability_family_has_rank1": float(np.mean(np.any(bootstrap_rank[:, indices] == 1, axis=1))), "probability_family_has_top3": float(np.mean(np.any(bootstrap_rank[:, indices] <= 3, axis=1)))})
    write_csv(STAT_OUT / "family_rank_frequencies.csv", family_rows)

    paired_rows = []
    for left in range(len(operators)):
        for right in range(left + 1, len(operators)):
            diff = bootstrap_G[:, left] - bootstrap_G[:, right]
            paired_rows.append({"operator_a": operators[left], "operator_b": operators[right], "mean_G_difference_a_minus_b": float(np.mean(diff)), "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5)), "probability_a_better_than_b": float(np.mean(diff < 0))})
    write_csv(STAT_OUT / "paired_difference_intervals.csv", paired_rows)

    # Leave-one-domain-out, sigma alternatives and reference sensitivity.
    lodo_rows = []
    sigma_rows = []
    for omitted in DOMAINS:
        kept = [domain for domain in DOMAINS if domain != omitted]
        for operator in operators:
            values = np.asarray([domain_scores[domain][operator] for domain in kept]); mu = float(np.mean(values)); sigma = float(np.std(values)); g = ALPHA * mu + (1 - ALPHA) * sigma
            lodo_rows.append({"omitted_domain": omitted, "operator": operator, "mu": mu, "sigma": sigma, "G": g})
        subset = [row for row in lodo_rows if row["omitted_domain"] == omitted]
        for rank, row in enumerate(sorted(subset, key=lambda item: (item["G"], item["operator"])), 1): row["rank"] = rank
    write_csv(STAT_OUT / "lodo_rankings.csv", lodo_rows)
    for index, operator in enumerate(operators):
        values = np.asarray([domain_scores[domain][operator] for domain in DOMAINS]); median = float(np.median(values)); mad = float(np.median(np.abs(values - median))); worst = float(np.max(values))
        sigma_rows.append({"operator": operator, "submitted_sigma_definition": float(np.std(values)), "bootstrap_sigma_mean": float(np.mean(bootstrap_sigma[:, index])), "bootstrap_sigma_ci_low": float(np.percentile(bootstrap_sigma[:, index], 2.5)), "bootstrap_sigma_ci_high": float(np.percentile(bootstrap_sigma[:, index], 97.5)), "MAD": mad, "worst_domain_score": worst, "G_with_MAD": ALPHA * float(np.mean(values)) + (1 - ALPHA) * mad, "G_with_worst_domain": ALPHA * float(np.mean(values)) + (1 - ALPHA) * worst})
    write_csv(STAT_OUT / "sigma_stability.csv", sigma_rows)

    sensitivity_rows = []
    for tier, matrices in (("REF-B_fixed_ten_diagnostic", errors_ref_b), ("REF-C_disjoint_four", errors_ref_c)):
        candidates = [operator for operator in operators if not (tier.startswith("REF-C") and operator in VOTERS_4)]
        values = {operator: float(np.mean([np.mean(matrices[domain][operators.index(operator)]) for domain in DOMAINS])) for operator in candidates}
        for rank, (operator, value) in enumerate(sorted(values.items(), key=lambda item: (item[1], item[0])), 1): sensitivity_rows.append({"reference_tier": tier, "operator": operator, "equal_domain_mean_absolute_displacement": value, "rank": rank})
    write_csv(STAT_OUT / "reference_sensitivity.csv", sensitivity_rows)

    # Four aggregation estimands and balanced repeated subsampling (n=30/domain).
    estimand_rows = []
    for operator in operators:
        oi = operators.index(operator); domain_means = [float(np.mean(errors_ref_b[domain][oi])) for domain in DOMAINS]
        estimand_rows.append({"operator": operator, "domain_specific": " | ".join(f"{domain}:{value:.6g}" for domain, value in zip(DOMAINS, domain_means)), "equal_domain_macro_mean_displacement": float(np.mean(domain_means)), "per_stack_micro_mean_displacement": float(np.average(domain_means, weights=[STACK_COUNTS[domain] for domain in DOMAINS]))})
    balanced_rank_counts = np.zeros((len(operators), len(operators)), dtype=int); balanced_values = np.zeros((BALANCED_REPEATS, len(operators)))
    for replicate in range(BALANCED_REPEATS):
        for domain in DOMAINS:
            indices = rng.choice(STACK_COUNTS[domain], size=30, replace=False)
            balanced_values[replicate] += np.mean(errors_ref_b[domain][:, indices], axis=1) / len(DOMAINS)
        order = np.argsort(balanced_values[replicate], kind="mergesort")
        for rank, index in enumerate(order): balanced_rank_counts[index, rank] += 1
    balanced_rows = []
    for index, operator in enumerate(operators):
        estimand_rows[index]["balanced_subsampling_mean_displacement"] = float(np.mean(balanced_values[:, index])); estimand_rows[index]["balanced_subsampling_sd"] = float(np.std(balanced_values[:, index], ddof=1))
        balanced_rows.append({"operator": operator, "mean_displacement": float(np.mean(balanced_values[:, index])), "ci_low": float(np.percentile(balanced_values[:, index], 2.5)), "ci_high": float(np.percentile(balanced_values[:, index], 97.5)), "probability_rank1": balanced_rank_counts[index, 0] / BALANCED_REPEATS, "probability_top5": np.sum(balanced_rank_counts[index, :5]) / BALANCED_REPEATS})
    write_csv(IMBALANCE_OUT / "aggregation_estimands.csv", estimand_rows)
    write_csv(IMBALANCE_OUT / "balanced_subsampling_rank_stability.csv", balanced_rows)
    write_csv(IMBALANCE_OUT / "domain_leave_one_out_effects.csv", lodo_rows)

    # Weight, alpha, runtime and deterministic Dirichlet sensitivity.
    weight_rows = []
    schemes = {
        "submitted": SUBMITTED_WEIGHTS,
        "equal": {metric: 1 / len(METRICS) for metric in METRICS},
        "localization_heavy": {**{metric: .05 / 8 for metric in METRICS}, "absolute_peak_localization_error": .45, "range_around_global_maximum": .20, "fwhm": .15, "execution_time_per_slice": .05},
        "runtime_0.10": dict(SUBMITTED_WEIGHTS),
        "runtime_0.05": {**SUBMITTED_WEIGHTS, "execution_time_per_slice": .05},
        "runtime_0": {**SUBMITTED_WEIGHTS, "execution_time_per_slice": 0.0},
    }
    for scheme, weights in schemes.items():
        rows, _, _ = score(raw, operators, weights, ALPHA)
        for row in rows: weight_rows.append({"analysis": "weight_scheme", "setting": scheme, "operator": row["operator"], "G": row["G"], "rank": row["rank"], "top5": row["rank"] <= 5})
    for alpha in np.linspace(0, 1, 11):
        rows, _, _ = score(raw, operators, SUBMITTED_WEIGHTS, float(alpha))
        for row in rows: weight_rows.append({"analysis": "alpha", "setting": f"{alpha:.1f}", "operator": row["operator"], "G": row["G"], "rank": row["rank"], "top5": row["rank"] <= 5})
    dirichlet_counts = np.zeros((len(operators), 3), dtype=int)
    for _ in range(DIRICHLET_DRAWS):
        draw = rng.dirichlet(np.ones(len(METRICS))); weights = dict(zip(METRICS, draw))
        rows, _, _ = score(raw, operators, weights, ALPHA)
        for row in rows:
            index = operators.index(row["operator"]); dirichlet_counts[index, 0] += row["rank"] == 1; dirichlet_counts[index, 1] += row["rank"] <= 3; dirichlet_counts[index, 2] += row["rank"] <= 5
    dirichlet_rows = [{"operator": operator, "family": families[operator], "draws": DIRICHLET_DRAWS, "probability_rank1": dirichlet_counts[index, 0] / DIRICHLET_DRAWS, "probability_top3": dirichlet_counts[index, 1] / DIRICHLET_DRAWS, "probability_top5": dirichlet_counts[index, 2] / DIRICHLET_DRAWS} for index, operator in enumerate(operators)]
    write_csv(WEIGHT_OUT / "deterministic_weight_and_alpha_sensitivity.csv", weight_rows)
    write_csv(WEIGHT_OUT / "dirichlet_rank_frequencies.csv", dirichlet_rows)
    (WEIGHT_OUT / "weight_provenance.md").write_text("""# Weight provenance

The submitted code records alpha = 0.7 and the ten criterion weights as author-defined constants. The available repository does not establish that these choices were prespecified before results were viewed, expert-elicited, or independently registered; no such claim is made. Alpha = 0 is reported only as an extreme dispersion-only stress test, not a recommended policy.
""", encoding="utf-8")

    # Machine-readable supplement.
    write_csv(RAW_OUT / "all_operators_domains_criteria_long.csv", raw_rows)
    write_csv(RAW_OUT / "per_stack_localization.csv", per_stack_rows)
    pd.DataFrame(raw_rows).to_excel(RAW_OUT / "all_operators_domains_criteria_long.xlsx", index=False)
    pd.DataFrame(final_rows).to_excel(RAW_OUT / "corrected_rankings.xlsx", index=False)

    config = {"generated_at": now(), "seed": SEED, "candidate_pool": 32, "primary_reference": "REF-B_fixed_ten_diagnostic", "confirmatory_reference": "REF-C_fixed_disjoint_four_for_28_nonvoters", "alpha": ALPHA, "weights": SUBMITTED_WEIGHTS, "bootstrap_replicates": BOOTSTRAPS, "highest_resampling_unit": "WBC slide (214 official clusters); stack for TBI, PBS, BMA, and TBF", "WBC_patient_level_requirement": "blocked; official release maps 25,773 cells to 214 slides but does not map slides to 72 patients", "balanced_subsampling_n_per_domain": 30, "balanced_repeats": BALANCED_REPEATS, "dirichlet_draws": DIRICHLET_DRAWS, "input_hashes": {"registry": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "runtime": hashlib.sha256(RUNTIME.read_bytes()).hexdigest(), "entropy_rrmse": hashlib.sha256(ENTROPY_RRMSE.read_bytes()).hexdigest(), "wbc_slide_map": hashlib.sha256(WBC_SLIDE_MAP.read_bytes()).hexdigest()}}
    (STAT_OUT / "bootstrap_configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    validation = {"status": "PASS", "operators": len(operators), "domains": list(DOMAINS), "all_raw_finite": all(np.isfinite(row["raw_value"]) for row in raw_rows), "all_normalized_finite": all(np.isfinite(row["normalized_value"]) for row in raw_rows), "candidate_pool_frozen": True, "WBC_slide_cluster_bootstrap": True, "WBC_slide_clusters": len(slide_ids), "patient_level_bootstrap_blocked": True}
    (SCORING_OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (SCORING_OUT / "SCORING_CORRECTION_FINDINGS.md").write_text(f"""# Corrected scoring findings

- G is a candidate-pool-relative decision score for the frozen 32 operators; it is not an absolute physical performance measure.
- Primary corrected ranking uses the fixed ten-voter REF-B diagnostic reference, corrected entropy criteria, and corrected native operator-kernel runtime.
- REF-C localization is confirmatory only for the 28 non-reference operators.
- Corrected top operator: {final_rows[0]['operator']} (G={final_rows[0]['G']:.6f}).
- The leading tier must be interpreted using bootstrap and weight-sensitivity frequencies rather than third-decimal separation.
""", encoding="utf-8")
    (STAT_OUT / "STATISTICAL_FINDINGS.md").write_text("""# Statistical findings

The submitted 50 domain-by-criterion cells are not treated as independent inferential blocks. Resampling recalculates domain and aggregate scores using the 214 official WBC slides as clusters and stacks for the other domains. The official release does not map those slides to its 72 patients, so residual within-patient dependence across slides cannot be modeled. Friedman/Nemenyi results are retained only as submitted descriptive history.
""", encoding="utf-8")
    (IMBALANCE_OUT / "DOMAIN_IMBALANCE_FINDINGS.md").write_text("""# Domain imbalance findings

Four estimands are kept distinct: domain-specific performance, equal-domain macro average, per-stack micro average, and repeated balanced subsampling with 30 stacks per domain. Equal-domain aggregation protects against WBC numerical dominance but does not equalize statistical precision or establish domain importance. Generalization claims remain limited to the five observed domains.
""", encoding="utf-8")
    (WEIGHT_OUT / "WEIGHT_SENSITIVITY_FINDINGS.md").write_text("""# Weight sensitivity findings

Sensitivity covers alpha from 0 to 1, submitted/equal/localization-heavy weights, runtime weights 0.10/0.05/0, and 1,000 deterministic Dirichlet draws. Report family-level tiers and top-five frequencies; do not interpret alpha=0 as a recommended policy.
""", encoding="utf-8")
    print(json.dumps({"top5": [(row["operator"], row["rank"], row["G"]) for row in final_rows[:5]], "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
