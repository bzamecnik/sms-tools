cdef extern from "twm.h":
	
	int TWM_C(double *pfreq, double *pmag, int nPeaks, double *f0c, int nf0c, double *f0, double *f0error)
