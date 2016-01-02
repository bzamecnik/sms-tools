// Spectral synthesis of sinusoids.

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#ifndef SPEC_SYNTH_H

#define SPEC_SYNTH_H

#define BH_SIZE 1001
#define BH_SIZE_BY2 501

void genbh92lobe_C(double *x, double *y, int N);
void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec);

#endif  //SPEC_SYNTH_H