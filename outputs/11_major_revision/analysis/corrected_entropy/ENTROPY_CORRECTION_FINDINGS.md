# Entropy correction findings

- The submitted implementation excluded TBF values above 256 after converting uint16 images to float.
- Submitted TBF constant curves: 182 / 182.
- Corrected TBF constant curves: 0 / 182.
- TBF stacks with a changed entropy peak: 182 / 182.
- All five domains were recomputed under the same fixed definition; see the domain summary and validation file.
- Downstream reference labels must use the corrected curves generated here.

Elapsed: 392.7 seconds.
