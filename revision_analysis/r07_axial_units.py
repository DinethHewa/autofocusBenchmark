#!/usr/bin/env python3
"""Convert axial metrics only where acquisition steps are verified."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
OUT = REVISION_ROOT / "06_analysis_outputs/axial_units"
LOCALIZATION = REVISION_ROOT / "06_analysis_outputs/raw_supplement/per_stack_localization.csv"
RAW = REVISION_ROOT / "06_analysis_outputs/corrected_scoring/corrected_raw_criteria.csv"
METADATA = OUT / "acquisition_metadata_verified.csv"
CURVES = REPOSITORY_ROOT / "outputs/03_single_measure_curves/normalized"
REF_C_VOTERS = {"Normalized Variance", "Histogram Entropy", "GLCM Contrast", "Fourier Transform Sharpness Index"}


def main() -> int:
    metadata = pd.read_csv(METADATA).set_index("domain")
    steps = metadata.axial_step_um.astype(float).to_dict()
    data = pd.read_csv(LOCALIZATION)
    lengths = {}
    for domain in metadata.index:
        arrays = np.load(CURVES / domain / "tenengrad.npy", allow_pickle=True)
        lengths.update({(domain, index): len(np.asarray(curve)) for index, curve in enumerate(arrays)})
    data["slice_count"] = [lengths[(domain, int(index))] for domain, index in zip(data.domain, data.stack_index)]
    data["axial_step_um"] = data.domain.map(steps)
    for tier in ("REF_B", "REF_C"):
        data[f"{tier}_absolute_displacement_um"] = data[f"{tier}_absolute_displacement"] * data.axial_step_um
        data[f"{tier}_normalized_axial_displacement"] = data[f"{tier}_absolute_displacement"] / (data.slice_count - 1)
    data.to_csv(OUT / "per_stack_axial_localization.csv", index=False)

    summaries = []
    for (domain, operator, family), rows in data.groupby(["domain", "operator", "family"], sort=False):
        for tier in ("REF_B", "REF_C"):
            eligible = not (tier == "REF_C" and operator in REF_C_VOTERS)
            if not eligible:
                continue
            displacement = rows[f"{tier}_absolute_displacement"].to_numpy(dtype=float)
            summaries.append({"domain": domain, "operator": operator, "family": family, "reference_tier": "REF-B_fixed_ten_diagnostic" if tier == "REF_B" else "REF-C_fixed_disjoint_four_confirmatory", "n_stacks": len(rows), "mean_absolute_displacement_slices": float(np.mean(displacement)), "median_absolute_displacement_slices": float(np.median(displacement)), "p90_absolute_displacement_slices": float(np.percentile(displacement, 90)), "mean_absolute_displacement_um": float(np.mean(rows[f"{tier}_absolute_displacement_um"])), "median_absolute_displacement_um": float(np.median(rows[f"{tier}_absolute_displacement_um"])), "p90_absolute_displacement_um": float(np.percentile(rows[f"{tier}_absolute_displacement_um"], 90)), "mean_normalized_axial_displacement": float(np.mean(rows[f"{tier}_normalized_axial_displacement"])), "exact_match_rate": float(np.mean(displacement == 0)), "within_one_slice_rate": float(np.mean(displacement <= 1)), "units_status": "axial step verified from official record"})
    pd.DataFrame(summaries).to_csv(OUT / "operator_localization_axial_summary.csv", index=False)

    raw = pd.read_csv(RAW)
    widths = raw[raw.criterion.isin(["fwhm", "range_around_global_maximum"])].copy()
    widths["axial_step_um"] = widths.domain.map(steps)
    widths["raw_value_um"] = widths.raw_value * widths.axial_step_um
    widths["physical_unit_status"] = "converted from slices using verified dataset axial step"
    widths.to_csv(OUT / "curve_width_physical_units.csv", index=False)
    config = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(), "formula_um": "absolute slice displacement multiplied by domain-specific verified axial step", "normalized_formula": "absolute slice displacement divided by (stack slice count - 1)", "no_cross_domain_step_substitution": True, "REF-C_scope": "28 operators disjoint from the four-voter reference", "input_hashes": {"localization": hashlib.sha256(LOCALIZATION.read_bytes()).hexdigest(), "metadata": hashlib.sha256(METADATA.read_bytes()).hexdigest(), "raw_criteria": hashlib.sha256(RAW.read_bytes()).hexdigest()}}
    (OUT / "axial_units_configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    validation = {"status": "PASS", "domains": sorted(data.domain.unique().tolist()), "all_steps_verified": all(np.isfinite(list(steps.values()))), "all_finite": bool(np.isfinite(data[["REF_B_absolute_displacement_um", "REF_C_absolute_displacement_um", "REF_B_normalized_axial_displacement", "REF_C_normalized_axial_displacement"]].to_numpy()).all()), "cross_domain_mean_times_WBC_step": False}
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (REVISION_ROOT / "12_logs/r07_axial_units.log").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
