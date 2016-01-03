# Native Cython extensions

There are two modules written in C and wrapped via Cython:

- `spec_synth` - synthesis of sinusoids in the spectral domain
- `twm` - [Two-Way Mismatch algorithm](http://ems.music.uiuc.edu/beaucham/papers/JASA.04.94.pdf)
  for fundamental frequency estimation in pitch tracking

The reasons for writing the code in C rather than Python were poor performance
of the original code. We should measure the performance of both implementations
and reconsider whether native code is really needed given the more complicated
build process and problems with generating API documentation.
