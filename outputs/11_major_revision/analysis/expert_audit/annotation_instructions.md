# Blinded expert annotation instructions

Run `python blinded_annotation_package/serve_annotations.py` from the expert-audit directory in an environment containing OpenCV and NumPy, then open the printed local URL. Each assessor works independently with a unique ID. Scroll every z-slice, select one best-focus slice, optionally enter an inclusive acceptable-focus interval, and mark uncertain or ungradable when appropriate. Do not consult algorithm curves, consensus indices, or another assessor. A third adjudicator reviews disagreements only after both assessor files are locked.

The primary sample is the inferential set. The disagreement-enriched set is qualitative and must never be pooled into unbiased agreement estimates. Within-stack display normalization is fixed across all slices. Raw images are served read-only from their source locations and are not copied into this package.
