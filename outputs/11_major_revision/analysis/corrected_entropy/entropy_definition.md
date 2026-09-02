# Corrected Histogram Entropy definition

Each integer image is mapped from the full representable range of its stored dtype to 256 fixed equal-width bins. Thus uint8 uses [0,255] and uint16 uses [0,65535], with endpoints included. Floating images are interpreted in a declared fixed range, [0,1] by default. The method performs no per-image min-max normalization. Every finite pixel contributes to exactly one bin, probabilities are histogram counts divided by the pixel count, and entropy is reported in bits as `-sum(p * log2(p))` over non-zero bins.

This convention is invariant for equivalent uint8 content encoded as uint16 by multiplication by 257 and prevents the submitted uint16 exclusion defect.
