# CFM4 Diagnostic Interpretation

The reviewer concern that CFM4 collapses to FTSI is not literally correct at the implementation level. The protected square root is `psqrt(x) = sqrt(abs(x) + eps)`, not zero-clipping. Therefore, slices with BG < GSE still contribute through the absolute-value branch. Across datasets, BG - GSE was negative for a mean fraction of 0.458 of evaluated slices.

The normalized CFM4 and FTSI curves are highly similar by correlation and peak agreement; at the curve level, the incremental composite contribution is therefore weak/modest rather than a clearly distinct focus response. The mean dataset-level Pearson correlation between normalized CFM4 and FTSI curves was 0.955, and the mean exact peak-agreement rate was 0.954.

In common scoring, CFM4 had value score 0.1623 with value rank 11, and rank score 15.7379 with rank rank 1. FTSI had value score 0.1409 with value rank 8, and rank score 20.1508 with rank rank 24. CFM4's advantage over FTSI is score-dependent rather than uniformly value-dominant. Thus, the rank-level fusion result is supported, but the value-level contribution over standalone FTSI should be described as limited rather than value-dominant.

The internal terminal name `Curvelet Transform Sharpness Index` is a naming legacy; the current implementation maps it to Wavelet Detail Energy (db1). Manuscript text should use WDE_db1 or explicitly disclose the internal-name correction.
