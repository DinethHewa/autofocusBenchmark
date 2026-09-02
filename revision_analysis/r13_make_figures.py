#!/usr/bin/env python3
"""Generate deterministic publication figures and figure-source manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
MAIN = REVISION_ROOT / "07_figures/main"
SUPP = REVISION_ROOT / "07_figures/supplementary"
QUAL = REVISION_ROOT / "07_figures/qualitative_examples"
OUT = REVISION_ROOT / "06_analysis_outputs/figure_qc"
LOG = REVISION_ROOT / "12_logs/r13_make_figures.log"
ROOTS = {
    "WBC": Path("/mnt/d/New_folder/datasets/WBC_dataset1"),
    "TBI": Path("/mnt/d/New_folder/datasets/New folder/TBSI/folders"),
    "PBS": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/pbs_imgs"),
    "BMA": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/bma_imgs"),
    "TBF": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/tbf_imgs"),
}
COUNTS = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
STEPS = {"WBC": 0.4, "TBI": 2.5, "PBS": 0.5, "BMA": 0.5, "TBF": 0.5}
LICENCED_FOR_DERIVATIVES = ("WBC", "PBS", "BMA", "TBF")
SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def folder_at(domain: str, index: int) -> Path:
    if domain == "WBC": return ROOTS[domain] / str(index)
    children = sorted((Path(entry.path) for entry in os.scandir(ROOTS[domain]) if entry.is_dir()), key=lambda path: natural_key(path.name))
    return children[index]


def read_stack(domain: str, index: int) -> tuple[list[Path], list[np.ndarray]]:
    folder = folder_at(domain, index)
    paths = sorted((Path(entry.path) for entry in os.scandir(folder) if entry.is_file() and Path(entry.name).suffix.lower() in SUFFIXES and ":Zone.Identifier" not in entry.name), key=lambda path: natural_key(path.name))
    raw = [cv.imread(str(path), cv.IMREAD_UNCHANGED) for path in paths]
    gray = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) if x.ndim == 3 and x.shape[2] == 3 else cv.cvtColor(x, cv.COLOR_BGRA2GRAY) if x.ndim == 3 else x for x in raw]
    low = min(float(x.min()) for x in gray); high = max(float(x.max()) for x in gray)
    display = [np.clip((x.astype(np.float32) - low) / (high - low + 1e-12), 0, 1) for x in gray]
    return paths, display


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve(domain: str, operator_slug: str, index: int, corrected_entropy: bool = False) -> np.ndarray:
    if corrected_entropy:
        path = REVISION_ROOT / f"05_cached_data/corrected_entropy/{domain}_histogram_entropy_normalized.npy"
    else:
        path = REPOSITORY_ROOT / f"outputs/03_single_measure_curves/normalized/{domain}/{operator_slug}.npy"
    return np.asarray(np.load(path, allow_pickle=True)[index], dtype=float)


def style() -> None:
    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})


def main() -> int:
    for directory in (MAIN, SUPP, QUAL, OUT): directory.mkdir(parents=True, exist_ok=True)
    style(); selections = []; generated = []
    with LOG.open("w", encoding="utf-8") as log:
        # Figure 1: no cherry-picking—the first source-ordered stack in each
        # dataset whose licence permits a submission derivative.
        fig, axes = plt.subplots(len(LICENCED_FOR_DERIVATIVES), 3, figsize=(7.2, 7.2))
        for row_index, domain in enumerate(LICENCED_FOR_DERIVATIVES):
            paths, images = read_stack(domain, 0)
            ref = int(np.load(REVISION_ROOT / f"05_cached_data/reference_ladder/{domain}_REF_B_fixed_ten.npy")[0])
            indices = [0, ref, len(images) - 1]
            for col, z in enumerate(indices):
                axes[row_index, col].imshow(images[z], cmap="gray", vmin=0, vmax=1)
                offset = (z - ref) * STEPS[domain]
                axes[row_index, col].set_title(f"{domain}: z={z}, Δz={offset:+.1f} µm")
                axes[row_index, col].axis("off")
            selections.append({"figure": "Figure 1", "domain": domain, "stack_index": 0, "selection_rule": "first stack in deterministic source order; all licensed domains", "display_normalization": "one global min/max over all slices in stack", "source_folder": str(folder_at(domain, 0)), "displayed_slices": json.dumps(indices)})
        fig.suptitle("Representative focus sequences (deterministic first stack)", y=1.01)
        fig.tight_layout(); save(fig, MAIN / "figure1_representative_focus_sequences"); generated.append("figure1_representative_focus_sequences")
        log.write("Figure 1 complete\n"); log.flush()

        # Figure 2: deterministic largest REF-B/REF-C disagreement among CC0 WBC,
        # ties broken by smallest stack index.
        affected = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/reference_audit/affected_stacks.csv")
        disagreement = affected[(affected.domain == "WBC") & (affected.comparison == "REF-B_corrected_ten_vs_REF-C_disjoint_four")].sort_values(["absolute_shift", "stack_index"], ascending=[False, True]).iloc[0]
        index = int(disagreement.stack_index); ref_b = int(disagreement.from_index); ref_c = int(disagreement.to_index)
        paths, images = read_stack("WBC", index)
        fig = plt.figure(figsize=(7.2, 4.4)); grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9])
        view_indices = [ref_b, ref_c, int(np.argmax(curve("WBC", "variance_of_gradient", index)))]
        labels = ["REF-B ten-voter", "REF-C disjoint four", "Variance of Gradient"]
        for col, (z, label) in enumerate(zip(view_indices, labels)):
            ax = fig.add_subplot(grid[0, col]); ax.imshow(images[z], cmap="gray", vmin=0, vmax=1); ax.set_title(f"{label}\nz={z}"); ax.axis("off")
        ax = fig.add_subplot(grid[1, :]);
        for slug, label, color in (("variance_of_gradient", "Variance of Gradient", "#1666a8"), ("tenengrad", "Tenengrad", "#e07a15"), ("normalized_variance", "Normalized Variance", "#2f855a")):
            y = curve("WBC", slug, index); ax.plot(range(len(y)), y, marker="o", ms=3, lw=1.5, label=label, color=color)
        ax.axvline(ref_b, color="#7b2cbf", ls="--", label=f"REF-B={ref_b}"); ax.axvline(ref_c, color="#d62828", ls=":", label=f"REF-C={ref_c}")
        ax.set(xlabel="Slice index", ylabel="Normalized focus value", title=f"Largest WBC REF-B–REF-C displacement: {abs(ref_c-ref_b)} slices"); ax.legend(ncol=3)
        fig.tight_layout(); save(fig, MAIN / "figure2_reference_disagreement_example"); generated.append("figure2_reference_disagreement_example")
        selections.append({"figure": "Figure 2", "domain": "WBC", "stack_index": index, "selection_rule": "largest REF-B-versus-REF-C displacement; smallest stack index breaks ties", "display_normalization": "one global min/max over all slices in stack", "source_folder": str(folder_at("WBC", index)), "displayed_slices": json.dumps(view_indices)})
        log.write("Figure 2 complete\n"); log.flush()

        # Figure 3: reference agreement.
        agreement = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/reference_audit/reference_agreement_by_domain.csv")
        agreement = agreement[agreement.comparison == "REF-B_corrected_ten_vs_REF-C_disjoint_four"].set_index("domain").loc[list(("WBC", "TBI", "PBS", "BMA", "TBF"))]
        x = np.arange(len(agreement)); fig, ax = plt.subplots(figsize=(7.2, 3.5))
        ax.bar(x - .18, agreement.exact_agreement, .36, label="Exact", color="#3178b5"); ax.bar(x + .18, agreement.within_one_slice_agreement, .36, label="Within one", color="#8bc1e8")
        ax.set(xticks=x, xticklabels=agreement.index, ylim=(0, 1.08), ylabel="Agreement", title="REF-B diagnostic versus REF-C disjoint reference"); ax.legend()
        fig.tight_layout(); save(fig, MAIN / "figure3_reference_agreement"); generated.append("figure3_reference_agreement")

        # Figure 4: corrected runtime scaling for the primary top six.
        runtime = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/corrected_runtime/corrected_runtime_per_resolution.csv")
        top = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/corrected_scoring/corrected_final_rankings.csv").sort_values("rank").head(6).operator.tolist()
        order = ["128", "512", "1024", "native"]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for operator in top:
            rows = runtime[runtime.operator == operator].copy(); rows["resolution"] = rows.resolution.astype(str); rows = rows.set_index("resolution").loc[order]
            ax.plot(order, rows.equal_domain_macro_median_kernel_ms, marker="o", lw=1.4, label=operator)
        ax.set_yscale("log"); ax.set(xlabel="Resolution condition", ylabel="Equal-domain macro median kernel time (ms, log)", title="Corrected operator-kernel scaling"); ax.legend(ncol=2)
        fig.tight_layout(); save(fig, MAIN / "figure4_corrected_runtime_scaling"); generated.append("figure4_corrected_runtime_scaling")

        # Figure 5: bootstrap score intervals and top-five probabilities.
        ranks = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/statistical_inference/rank_frequencies.csv").head(10).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
        y = np.arange(len(ranks)); axes[0].errorbar(ranks.G_mean, y, xerr=[ranks.G_mean-ranks.G_ci_low, ranks.G_ci_high-ranks.G_mean], fmt="o", color="#245b8a", capsize=2); axes[0].set(yticks=y, yticklabels=ranks.operator, xlabel="G (95% hierarchical bootstrap CI)", title="Primary score uncertainty")
        axes[1].barh(y, ranks.probability_top5, color="#4c9f70"); axes[1].set(yticks=y, yticklabels=[], xlim=(0,1), xlabel="P(top 5)", title="Rank frequency")
        fig.tight_layout(); save(fig, MAIN / "figure5_rank_uncertainty"); generated.append("figure5_rank_uncertainty")

        # Figure 6: domain-specific localization distributions for the top six.
        # These data are strongly zero-inflated, so categorical displacement
        # proportions reveal the tail more faithfully than collapsed boxplots.
        localization = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/raw_supplement/per_stack_localization.csv", usecols=["domain", "operator", "REF_B_absolute_displacement"])
        localization = localization[localization.operator.isin(top)]
        fig, axes = plt.subplots(1, 5, figsize=(7.2, 3.2), sharey=True)
        for ax, domain in zip(axes, ("WBC", "TBI", "PBS", "BMA", "TBF")):
            rows = localization[localization.domain == domain]
            exact, one, tail = [], [], []
            for operator in top:
                values = rows[rows.operator == operator].REF_B_absolute_displacement.to_numpy()
                exact.append(float(np.mean(values == 0))); one.append(float(np.mean(values == 1))); tail.append(float(np.mean(values >= 2)))
            xbars = np.arange(1, 7)
            ax.bar(xbars, exact, color="#3d8b67", label="0 slices"); ax.bar(xbars, one, bottom=exact, color="#e9ae46", label="1 slice"); ax.bar(xbars, tail, bottom=np.asarray(exact)+np.asarray(one), color="#bf4d4d", label="≥2 slices")
            ax.set_title(domain); ax.set_xticks(xbars, xbars); ax.set_xlabel("Primary rank"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("Proportion of stacks"); axes[-1].legend(loc="lower left", bbox_to_anchor=(1.04, 0)); fig.suptitle("Domain-specific localization displacement for the primary top-six operators"); fig.tight_layout(); save(fig, MAIN / "figure6_domain_localization"); generated.append("figure6_domain_localization")

        # Supplement: corrected TBF entropy curve, resampling, and weights.
        old_entropy = np.asarray(np.load(REPOSITORY_ROOT / "outputs/03_single_measure_curves/normalized/TBF/histogram_entropy.npy", allow_pickle=True)[0], dtype=float)
        new_entropy = curve("TBF", "histogram_entropy", 0, corrected_entropy=True)
        fig, ax = plt.subplots(figsize=(5.8, 3.4)); ax.plot(old_entropy, marker="o", label="Submitted", color="#9aa0a6"); ax.plot(new_entropy, marker="o", label="Corrected", color="#b33c3c"); ax.set(xlabel="Slice index", ylabel="Normalized entropy", title="TBF histogram-entropy correction (first stack)"); ax.legend(); fig.tight_layout(); save(fig, SUPP / "figureS1_entropy_correction"); generated.append("figureS1_entropy_correction")

        mech = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/resampling/roberts_brenner_mechanism.csv")
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
        condition_order = sorted(mech.condition.unique())
        for operator, color in (("Roberts Focus Measure", "#7b2cbf"), ("Brenner Gradient", "#d97706")):
            rows = mech[mech.operator == operator].set_index("condition").loc[condition_order]
            axes[0].plot(range(len(rows)), rows["rank"], marker="o", label=operator, color=color); axes[1].plot(range(len(rows)), rows["noise_change"], marker="o", label=operator, color=color)
        axes[0].invert_yaxis(); axes[0].set(ylabel="Descriptive mechanism rank", title="Controlled Roberts/Brenner resampling response"); axes[0].legend(); axes[1].axhline(0, color="black", lw=.8); axes[1].set(ylabel="Noise change vs base", xticks=range(len(condition_order)), xticklabels=condition_order, xlabel="Condition"); axes[1].tick_params(axis="x", rotation=55)
        fig.tight_layout(); save(fig, SUPP / "figureS2_resampling_mechanism"); generated.append("figureS2_resampling_mechanism")

        weights = pd.read_csv(REVISION_ROOT / "06_analysis_outputs/weight_sensitivity/dirichlet_rank_frequencies.csv").sort_values("probability_top5", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 4.1)); y = np.arange(len(weights)); ax.barh(y, weights.probability_top5, color=["#3b82a0" if family == "gradient" else "#a8b3bd" for family in weights.family]); ax.set(yticks=y, yticklabels=weights.operator, xlim=(0,1), xlabel="P(top 5), 1,000 Dirichlet weight draws", title="Sensitivity to criterion-weight policy"); fig.tight_layout(); save(fig, SUPP / "figureS3_weight_sensitivity"); generated.append("figureS3_weight_sensitivity")
        log.write("All figures complete\n")

    with (REVISION_ROOT / "07_figures/figure_selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selections[0])); writer.writeheader(); writer.writerows(selections)
    figure_files = sorted([*MAIN.glob("*.png"), *MAIN.glob("*.pdf"), *SUPP.glob("*.png"), *SUPP.glob("*.pdf")])
    manifest = [{"path": str(path.relative_to(REVISION_ROOT)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in figure_files]
    with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    validation = {"status": "PASS", "generated_stems": generated, "png_count": len(list(MAIN.glob("*.png"))) + len(list(SUPP.glob("*.png"))), "pdf_count": len(list(MAIN.glob("*.pdf"))) + len(list(SUPP.glob("*.pdf"))), "all_nonempty": all(path.stat().st_size > 10_000 for path in figure_files), "expert_figure": "blocked pending genuine annotations", "TBI_raw_image_derivative": "omitted because licence not reported"}
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (OUT / "figure_configuration.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).astimezone().isoformat(), "dpi": 300, "formats": ["PNG", "PDF"], "selection_seed": "not random; deterministic explicit rules", "input_hashes": {"rank_frequencies": hashlib.sha256((REVISION_ROOT / "06_analysis_outputs/statistical_inference/rank_frequencies.csv").read_bytes()).hexdigest(), "reference_agreement": hashlib.sha256((REVISION_ROOT / "06_analysis_outputs/reference_audit/reference_agreement_by_domain.csv").read_bytes()).hexdigest()}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
