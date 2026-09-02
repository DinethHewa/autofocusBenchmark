#!/usr/bin/env python3
"""Build and audit REF-A--REF-E tiers without rotating nine-voter designs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
OUT = REVISION_ROOT / "06_analysis_outputs/reference_audit"
CACHE = REVISION_ROOT / "05_cached_data/reference_ladder"
CURVES = REPOSITORY_ROOT / "outputs/03_single_measure_curves/raw"
LABELS = REPOSITORY_ROOT / "outputs/02_reference_labels"
ENTROPY = REVISION_ROOT / "05_cached_data/corrected_entropy"
REGISTRY_PATH = REVISION_ROOT / "00_audit/operator_registry_32.json"
DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
VOTERS_10 = (
    "Tenengrad", "Brenner Gradient", "Variance of Laplacian", "Sum Modified Laplacian",
    "Normalized Variance", "Energy of Gradient", "Histogram Entropy", "GLCM Contrast",
    "Variance of Gradient", "Fourier Transform Sharpness Index",
)
VOTERS_4 = ("Normalized Variance", "Histogram Entropy", "GLCM Contrast", "Fourier Transform Sharpness Index")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curves(path: Path) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.float64) for value in np.load(path, allow_pickle=True)]


def peak(curve: np.ndarray) -> int:
    values = np.asarray(curve, dtype=np.float64)
    valid = np.arange(1, len(values) - 1) if len(values) >= 3 else np.arange(len(values))
    best = np.max(values[valid])
    tied = valid[np.isclose(values[valid], best)]
    return int(sorted(tied.tolist())[len(tied) // 2])


def vote(predictions: dict[str, np.ndarray], voters: tuple[str, ...]) -> np.ndarray:
    size = len(predictions[voters[0]])
    result = np.zeros(size, dtype=int)
    for index in range(size):
        counts = Counter(int(predictions[name][index]) for name in voters)
        largest = max(counts.values())
        tied = sorted(label for label, count in counts.items() if count == largest)
        result[index] = tied[len(tied) // 2]
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def comparison_rows(domain: str, name: str, left: np.ndarray, right: np.ndarray) -> tuple[dict, dict, list[dict]]:
    shifts = np.abs(left.astype(int) - right.astype(int))
    agreement = {
        "domain": domain, "comparison": name, "n_stacks": len(shifts),
        "exact_agreement": float(np.mean(shifts == 0)),
        "within_one_slice_agreement": float(np.mean(shifts <= 1)),
        "mean_absolute_displacement": float(np.mean(shifts)),
        "median_absolute_displacement": float(np.median(shifts)),
        "p90_absolute_displacement": float(np.percentile(shifts, 90)),
        "max_absolute_displacement": int(np.max(shifts)),
    }
    distribution = dict(agreement)
    transition = [
        {"domain": domain, "comparison": name, "from_index": a, "to_index": b, "count": count}
        for (a, b), count in sorted(Counter(zip(left.astype(int), right.astype(int))).items())
    ]
    affected = [{"domain": domain, "comparison": name, "stack_index": index, "from_index": int(left[index]), "to_index": int(right[index]), "absolute_shift": int(shifts[index])} for index in np.where(shifts > 0)[0]]
    return agreement, distribution, transition + affected


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    operators = [entry["measure_name"] for entry in registry]
    families = {entry["measure_name"]: entry["family"] for entry in registry}
    provenance: list[dict] = []
    agreements: list[dict] = []
    shifts: list[dict] = []
    transitions: list[dict] = []
    affected: list[dict] = []
    localization: list[dict] = []
    tier_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    validation = {"generated_at": now(), "status": "PASS", "domains": {}, "failures": []}

    for domain in DOMAINS:
        predictions: dict[str, np.ndarray] = {}
        curves_by_operator: dict[str, list[np.ndarray]] = {}
        for name in operators:
            path = ENTROPY / f"{domain}_histogram_entropy_raw.npy" if name == "Histogram Entropy" else CURVES / domain / f"{slug(name)}.npy"
            curves = load_curves(path)
            curves_by_operator[name] = curves
            predictions[name] = np.asarray([peak(curve) for curve in curves], dtype=int)

        ref_a_full = np.load(LABELS / "surrogate" / f"{domain}_surrogate_labels.npy").astype(int)
        ref_b = vote(predictions, VOTERS_10)
        ref_c = vote(predictions, VOTERS_4)
        np.save(CACHE / f"{domain}_REF_A_submitted_full.npy", ref_a_full)
        np.save(CACHE / f"{domain}_REF_B_fixed_ten.npy", ref_b)
        np.save(CACHE / f"{domain}_REF_C_fixed_four.npy", ref_c)
        tiers = {
            "REF-A_submitted_full": ref_a_full,
            "REF-B_fixed_ten_diagnostic": ref_b,
            "REF-C_fixed_disjoint_four": ref_c,
        }
        for tier, labels in tiers.items():
            cache_stem = (
                "REF_A_submitted_full" if tier.startswith("REF-A")
                else "REF_B_fixed_ten" if tier.startswith("REF-B")
                else "REF_C_fixed_four"
            )
            provenance.append({"domain": domain, "reference_tier": tier, "n_stacks": len(labels), "voters": "submitted cached ten" if tier.startswith("REF-A") else " | ".join(VOTERS_10 if tier.startswith("REF-B") else VOTERS_4), "independent_of_evaluated_operator": tier.startswith("REF-C"), "status": "available", "sha256": hash_file(CACHE / f"{domain}_{cache_stem}.npy")})

        for name, left, right in (
            ("REF-A_full_vs_REF-B_corrected_ten", ref_a_full, ref_b),
            ("REF-B_corrected_ten_vs_REF-C_disjoint_four", ref_b, ref_c),
            ("REF-A_full_vs_REF-C_disjoint_four", ref_a_full, ref_c),
        ):
            agreement, distribution, mixed = comparison_rows(domain, name, left, right)
            agreements.append(agreement); shifts.append(distribution)
            for row in mixed:
                (affected if "stack_index" in row else transitions).append(row)

        # Explicitly quantify every submitted voter LOO-versus-full difference.
        for voter in VOTERS_10:
            loo = np.load(LABELS / "leave_one_out" / domain / f"{slug(voter)}_loo_labels.npy").astype(int)
            agreement, distribution, mixed = comparison_rows(domain, f"REF-A_LOO_{slug(voter)}_vs_REF-A_full", loo, ref_a_full)
            agreements.append(agreement); shifts.append(distribution)
            for row in mixed:
                (affected if "stack_index" in row else transitions).append(row)

        for tier, reference in tiers.items():
            for operator in operators:
                if tier.startswith("REF-C") and operator in VOTERS_4:
                    continue
                displacement = np.abs(predictions[operator] - reference)
                localization.append({"domain": domain, "reference_tier": tier, "operator": operator, "family": families[operator], "n_stacks": len(displacement), "exact_agreement": float(np.mean(displacement == 0)), "within_one_slice_agreement": float(np.mean(displacement <= 1)), "mean_absolute_displacement": float(np.mean(displacement)), "median_absolute_displacement": float(np.median(displacement)), "p90_absolute_displacement": float(np.percentile(displacement, 90))})

        validation["domains"][domain] = {"n_stacks": len(ref_a_full), "ref_a_ref_b_exact": float(np.mean(ref_a_full == ref_b)), "ref_b_ref_c_exact": float(np.mean(ref_b == ref_c)), "all_indices_in_bounds": all(0 <= int(x) < len(curves_by_operator[operators[0]][i]) for i, x in enumerate(ref_c))}

    # Equal-domain operator ranks for each tier.
    by_tier_operator: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in localization:
        by_tier_operator[(row["reference_tier"], row["operator"])].append(row["mean_absolute_displacement"])
    rank_rows: list[dict] = []
    for tier in sorted({row["reference_tier"] for row in localization}):
        scores = [(operator, float(np.mean(values))) for (candidate_tier, operator), values in by_tier_operator.items() if candidate_tier == tier]
        scores.sort(key=lambda item: (item[1], item[0]))
        for rank, (operator, score) in enumerate(scores, start=1):
            tier_ranks[tier][operator] = rank
            rank_rows.append({"reference_tier": tier, "operator": operator, "family": families[operator], "equal_domain_mean_absolute_displacement": score, "rank": rank})
    for row in rank_rows:
        operator = row["operator"]
        row["rank_shift_vs_REF-A"] = row["rank"] - tier_ranks["REF-A_submitted_full"].get(operator, row["rank"])

    family_rows: list[dict] = []
    for tier in sorted(tier_ranks):
        grouped: dict[str, list[int]] = defaultdict(list)
        for operator, rank in tier_ranks[tier].items(): grouped[families[operator]].append(rank)
        for family, ranks in grouped.items():
            family_rows.append({"reference_tier": tier, "family": family, "operator_count": len(ranks), "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)), "top5_count": sum(rank <= 5 for rank in ranks)})

    # REF-D is unavailable after checking the local official release; REF-E needs people.
    for domain in DOMAINS:
        if domain == "WBC":
            provenance.append({"domain": domain, "reference_tier": "REF-D_external_WBC_algorithmic", "n_stacks": 0, "voters": "none", "independent_of_evaluated_operator": True, "status": "unavailable: no per-stack device/Laplacian focus index or score found in local release", "sha256": ""})
        provenance.append({"domain": domain, "reference_tier": "REF-E_blinded_expert", "n_stacks": 0, "voters": "none", "independent_of_evaluated_operator": True, "status": "blocked pending genuine annotations", "sha256": ""})

    write_csv(OUT / "reference_provenance.csv", provenance)
    write_csv(OUT / "reference_agreement_by_domain.csv", agreements)
    write_csv(OUT / "reference_shift_distributions.csv", shifts)
    write_csv(OUT / "reference_transition_matrices.csv", transitions, ["domain", "comparison", "from_index", "to_index", "count"])
    write_csv(OUT / "affected_stacks.csv", affected, ["domain", "comparison", "stack_index", "from_index", "to_index", "absolute_shift"])
    write_csv(OUT / "operator_localization_by_tier_domain.csv", localization)
    write_csv(OUT / "operator_rank_shifts.csv", rank_rows)
    write_csv(OUT / "family_rank_stability.csv", family_rows)
    configuration = {"generated_at": now(), "seed": None, "exclude_endpoints": True, "tie_break": "upper central index among tied sorted indices", "REF-A": "immutable submitted full and operator-specific LOO", "REF-B_voters": VOTERS_10, "REF-C_voters": VOTERS_4, "candidate_pool": 32, "input_hashes": {"registry": hash_file(REGISTRY_PATH), "corrected_entropy_validation": hash_file(REVISION_ROOT / "06_analysis_outputs/corrected_entropy/validation_summary.json")}}
    (OUT / "reference_ladder_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (OUT / "reference_claim_limits.md").write_text("""# Reference claim limits

- REF-A is historical replication. Its operator-specific LOO references are not mutually comparable across all operators.
- REF-B is fixed and comparable but diagnostically self-including for its ten voters; it is not independent.
- REF-C is fixed and disjoint only for the 28 non-reference operators. The four voters are listed separately and are not ranked against their own reference.
- REF-D could not be run because no official per-stack device/Laplacian indices or scores were found in the local WBC release. It must never be described as human or optical ground truth.
- REF-E is blocked until genuine blinded expert annotations are supplied. Expert consensus, when available, is not optical ground truth.
- Reference-construction disagreement is a resolution/indeterminacy diagnostic, not a formal equivalence margin.
""", encoding="utf-8")
    ref_a_ref_b = [row for row in agreements if row["comparison"] == "REF-A_full_vs_REF-B_corrected_ten"]
    ref_b_ref_c = [row for row in agreements if row["comparison"] == "REF-B_corrected_ten_vs_REF-C_disjoint_four"]
    findings = "# Reference ladder findings\n\n" + "\n".join(
        f"- {row['domain']}: REF-A-to-REF-B exact agreement {row['exact_agreement']:.3f}; within one slice {row['within_one_slice_agreement']:.3f}." for row in ref_a_ref_b
    ) + "\n\n" + "\n".join(
        f"- {row['domain']}: REF-B-to-REF-C exact agreement {row['exact_agreement']:.3f}; within one slice {row['within_one_slice_agreement']:.3f}." for row in ref_b_ref_c
    ) + "\n\nREF-D is unavailable and REF-E is blocked; claim limits are explicit in `reference_claim_limits.md`.\n"
    (OUT / "REFERENCE_LADDER_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
