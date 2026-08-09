# Review-Response Results Summary

## Gradient-free voter sensitivity

Gradient-free surrogate labels were constructed using only Normalized Variance, Histogram Entropy, GLCM Contrast, and Fourier Transform Sharpness Index. No gradient or Laplacian voter was used. The rank-based top operator was Brenner Gradient, and the value-based top operator was Variance of Gradient. Gradient-family operators contributed 9 of the rank-based top 10 and 8 of the value-based top 10. This should be described as an internal surrogate-label sensitivity analysis, not as hardware ground truth.

## CFM4 vs FTSI diagnostic

The CFM4 diagnostic used the actual protected square root, `psqrt(x) = sqrt(abs(x) + eps)`. Therefore, BG < GSE does not zero out the correction term. The mean dataset-level CFM4-vs-FTSI Pearson correlation was 0.955, and the mean exact peak-agreement rate was 0.954. These values indicate that CFM4 is highly similar to FTSI at the curve/peak level, so the composite contribution should be framed as weak/modest. The rank-level fusion result can still be reported, but it should not be overclaimed as a clearly value-dominant improvement over standalone FTSI.

## Runtime-weight sensitivity

Runtime-weight sensitivity reused the saved raw metric tensors and did not rerun the benchmark. The default single-only top entity was Variance of Gradient, and the default union-pool top entity was Variance of Gradient. The accompanying sensitivity table reports whether reducing or removing runtime weight changes top-1, top-3, or top-5 conclusions.

## GP hyperparameter consistency

The GP audit found core recorded settings consistency = True. Full recorded-settings identity = False; if false, the difference is due to saved-result metadata fields that were added in later run-control patches, not to a rerun performed here.

## Formal metric definitions

Formal definitions for FWHM, curvature, steep-slope width, steep-to-gradual slope ratio, false maxima count, noise level, range around global maximum, and RRMSE under additive noise were written to the supplementary-methods folder. The docx export was not available because python-docx is not installed.

## Revised limitations paragraph

The revised manuscript should state that surrogate labels are consensus labels rather than optical ground truth; that the gradient-free analysis is a sensitivity analysis; that CFM4 is evaluated as an interpretable symbolic fusion but may remain curve-similar to FTSI; and that downstream/proxy analyses should not be described as diagnostic validation.
