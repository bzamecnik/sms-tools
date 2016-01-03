"""
Two-way mismatch algorithm for detection of fundamental frequency.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from twm cimport *

np.import_array()


def twm(pfreq, pmag, f0c):
    """
    Find fundamental frequency

    This is a native implementation of smst.utils.peaks.find_fundamental_twm_py.

    :param pfreq: peak frequencies in Hz
    :param pmag: peak magnitudes
    :param f0c: frequencies of f0 candidates
    :returns: f0, f0Error: fundamental frequency detected and its error
    """
    
    
    cdef np.ndarray[np.float_t, ndim=1] f0_arr
    cdef np.ndarray[np.float_t, ndim=1] f0Error_arr
    cdef np.ndarray[np.float_t, ndim=1] pfreq_arr
    cdef np.ndarray[np.float_t, ndim=1] pmag_arr
    cdef np.ndarray[np.float_t, ndim=1] f0c_arr
    
    f0_arr = np.ascontiguousarray(np.array([-1]), dtype=np.float)
    f0Error_arr = np.ascontiguousarray(np.array([-1]), dtype=np.float)

    pfreq_arr = np.ascontiguousarray(pfreq, dtype=np.float)
    pmag_arr = np.ascontiguousarray(pmag, dtype=np.float)
    f0c_arr = np.ascontiguousarray(f0c, dtype=np.float)

    TWM_C(<double*>pfreq_arr.data, <double *>pmag_arr.data, pfreq_arr.shape[0], <double *>f0c_arr.data, f0c_arr.shape[0], <double*>f0_arr.data, <double*>f0Error_arr.data)

    return f0_arr[0], f0Error_arr[0]
