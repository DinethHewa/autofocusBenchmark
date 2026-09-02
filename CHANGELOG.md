# Changelog

## 2.0.0 - 2026-09-02

Release accompanying the major revision of the manuscript (MDPI Journal of Imaging,
manuscript jimaging-4524210).

Corrections
- Reference construction is now described as it is executed: a plurality vote over
  per-voter peak indices with upper-middle tie-breaking and endpoint exclusion, not an
  arithmetic mean of normalized voter curves. The description was wrong; the analysis
  was not, and no result required recomputation.
- Histogram Entropy is binned on the source encoding range identified before
  floating-point conversion. The previous implementation applied a fixed range after
  conversion, so a 16-bit source contributed only its lowest 256 levels. Histogram
  Entropy is both a registered voter and a scored candidate, so reference indices and
  the ranking were regenerated from corrected curves.
- The runtime protocol is no longer described as single-threaded: the recorded OpenCV
  runtime reported 32 threads while the OMP, OpenBLAS, MKL and NumExpr variables were
  set to one. Both an equal-domain macro statistic and a plane-count-weighted micro
  statistic are now reported.

Method changes
- Three explicit reference constructions replace the rotating leave-one-operator-out
  design: REF-B (one fixed ten-voter consensus applied to all 32 candidates), REF-A
  (single-voter exclusion, index sensitivity only) and REF-C (four non-derivative
  voters, family-shift diagnostic).
- The 50 domain-by-criterion cells are no longer used for confirmatory inference.
  Uncertainty is estimated by 1,000 paired clustered bootstrap replicates, with the
  white-cell domain clustered by its 214 official slides.
- Four aggregation estimands are reported separately, with leave-one-domain-out.
- The image-resampling experiment is a one-factor-at-a-time design over 14 conditions,
  replacing the single square-grid condition that confounded scale, interpolation and
  aspect ratio.

Additions
- revision_analysis/            revision analysis scripts (r01 to r17) and their tests
- outputs/11_major_revision/    all revision analysis outputs, figures and configs

## 1.0.0 - 2026-06-22

- Initial reproducibility release for the smear-microscopy autofocus benchmark.
- Includes the 32-measure benchmark, LODO GP implementation, final all-dataset refit workflow, paper assets, and reviewer-response computations.
