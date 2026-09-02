# Runtime correction findings

- The submitted native column is an equal-domain macro-average, not a slice-weighted average over all evaluated slices.
- Submitted native and resized paths did not apply identical preprocessing/dtype conversion, and the submitted timing used no warm-up or repeated measurements.
- Corrected results separately report I/O, preprocessing, operator-kernel, and combined preprocessing-plus-kernel timing.
- All aggregation labels are explicit. The 10.7–24.4 ms submitted claim is retired and must not be reused.
- Final score recalculation must use the corrected native operator-kernel summary and separately test runtime weights 0.10, 0.05, and 0.
