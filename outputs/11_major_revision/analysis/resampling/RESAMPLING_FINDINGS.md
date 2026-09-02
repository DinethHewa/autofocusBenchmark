# Controlled resampling findings

This experiment supersedes the submitted forced 1024x1024 comparison. Native cached curves remain an observational comparator. The controlled mechanism experiment first forms one aspect-preserving 512-longest-side base per slice, then changes one factor at a time: scale (0.5x or 2x), interpolation (nearest, bilinear, bicubic, area), or square-grid distortion. All 32 frozen operators are present in every condition.

Roberts and Brenner changes are decomposed into localization, false-maxima, curve-noise and FWHM changes in `roberts_brenner_mechanism.csv`. Spearman correlations are labelled associations, not causal proof. A common physical-pixel-size experiment is unavailable because the required pixel-size metadata are not verified across domains.

The frozen supplement numbering itself was inconsistent with the manuscript list: S2 is alpha sensitivity, S5 is composite deduplication, and the native-versus-1024 rank-stability export is S7. The submitted 1024 result also confounded scale, aspect distortion and interpolation; it is retained only as historical evidence.
