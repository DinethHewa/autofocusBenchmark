# Smear-Microscopy Autofocus Focus-Measure Benchmark

Reproducibility repository for:

**Cross-Domain Benchmarking of Focus Measures for Smear-Microscopy Autofocus Under a Consensus-Audited Reference**

This repository contains the complete analysis code, normalized focus curves, aggregate
results, publication assets, and supporting computations. Raw microscopy images and the
46 GB derived stack cache are not redistributed.

## Study scope

- Five smear-microscopy domains: WBC, TBI, PBS, BMA, and TBF.
- 26,100 z-stacks evaluated at native resolution.
- Thirty-two implemented single focus measures.
- Ten evaluation criteria, including per-slice runtime.
- Consensus reference construction with leave-one-operator-out auditing and an
  independent four-voter non-derivative reconstruction.
- Leave-one-domain-out genetic programming with ten seeds per fold.
- Ten-seed final all-domain refit.
- Fourteen retained composite candidates evaluated under common scoring.

## Repository structure

```text
config/                 Experiment settings, paths, weights, and asset registry
scripts/                Pipeline stages 00-11
src/                    Measures, labels, evaluation, GP, plotting, and utilities
outputs/
  01_stacks/metadata/   Stack metadata only; image arrays are excluded
  02_reference_labels/  Source/surrogate labels and provenance manifests
  03_single_measure_curves/
    normalized/         Saved normalized curves for all 32 measures
    timing/             Saved timing records
  04_single_measure_eval/
  05_gp_runs/           Compact GP summaries and deduplicated expressions
  06_composite_eval/
  07_statistics/
  09_paper/             Final tables, figures, captions, and manifests
  10_review_response_computations/
docs/                   Reproduction, data, output, and release instructions
```

The corresponding Zenodo archive contains raw focus curves and complete per-seed GP
artifacts in addition to the files provided here.

## Installation

Recommended:

```bash
conda env create -f environment.yml
conda activate smear-focus
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA/CuPy is optional. Production GP results in the archive used the CPU backend.

## Reproduction

Full pipeline:

```bash
python scripts/11_run_full_pipeline.py --full-run --skip-downstream
```

The production GP stage is computationally expensive. Existing results are included so
inspection and downstream reporting do not require rerunning GP.

Supporting computations from saved normalized curves:

```bash
python tools/review_response/run_review_response_computations.py
```

Validate the release structure and included derived outputs:

```bash
python tools/validate_release.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for staged commands and checkpoint
behavior.

## Interpretation notes

- Reference indices are consensus-derived references, not hardware-calibrated optical
  ground truth. Reported peak-localization values are errors relative to that reference.
- The optional downstream analysis is a focus-quality proxy, not diagnostic validation.
- Single-only and union value scores use different min-max normalization pools and are
  not numerically interchangeable.

## Citation

If you use this software or its archived outputs, cite the associated manuscript and this
record. Machine-readable metadata is provided in `CITATION.cff` and `.zenodo.json`.

## Licenses

- Source code: MIT License, see `LICENSE`.
- Repository-authored tables, figures, and documentation: CC BY 4.0, see `OUTPUTS_LICENSE.md`.
- No rights are granted to the source microscopy datasets, which are not included.

## Author

Dineth Hewavitharana (https://orcid.org/0009-0000-2439-759X)
Department of Mechanical Engineering, University of Moratuwa, Katubedda 10400, Sri Lanka
