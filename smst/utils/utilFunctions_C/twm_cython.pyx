#this is a cython wrapper on C functions to call them in python

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from twm cimport *

np.import_array()


def twm(pfreq, pmag, f0c):
    """This is a cython wrapper for a C function which is bit exact with the python version of this function
       For information about the input arguments please refere to the original python function
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
