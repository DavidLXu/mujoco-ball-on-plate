# Source image

`fourier_wiki.jpg` is the public-domain engraving **“Joseph Fourier”**, by
A. F. B. Geille after J. Boilly.  Source:

https://commons.wikimedia.org/wiki/File:Joseph_Fourier.jpg

The engraving is in the public domain (Fourier died in 1830; the print is a
19th-century work).

`fourier_portrait_points.npy` is the closed outer-silhouette contour extracted
from that image by `../extract_portrait.py`.  Only the outer silhouette is used:
the ball traces one continuous closed stroke, so the engraving's internal
hatching and facial details are not reproduced.

`demo_star.png` is a synthetic transparent test image used by the pipeline demo and tests.
