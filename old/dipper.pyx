import numpy as np
from scipy.ndimage import minimum_filter1d

cdef extern from "math.h":
    double sqrt(double m)

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# def group_observations(double[:] mjd, double[:] mag, double[:] magerr):
def group_observations(np.ndarray[np.float64_t] mjd, np.ndarray[np.float64_t] mag,
                       np.ndarray[np.float64_t] magerr):
    """Group observations that are on the same night"""
    # Figure out how long the output array should be
    cdef int lastval = -1
    cdef int num_nights = 0
    cdef int num_obs = len(mjd)
    for i in range(num_obs):
        if <int>mjd[i] != lastval:
            lastval = <int>mjd[i]
            num_nights += 1
        
    cdef np.ndarray[np.float64_t, ndim=1] out_mjd = np.empty(num_nights)
    cdef np.ndarray[np.float64_t, ndim=1] out_mag = np.empty(num_nights)
    cdef np.ndarray[np.float64_t, ndim=1] out_magerr = np.empty(num_nights)
    
    cdef int current_base_mjd = -1
    
    # Implement a weighted mean on both mag and mjd.
    cdef int out_idx = 0
    cdef double mjd_num = 0
    cdef double mag_num = 0
    cdef double denom = 0
    cdef double inv_var
    
    for idx in range(num_obs + 1):
        # Record everything if we are at the end of a night.
        if current_base_mjd > 0 and (idx == num_obs or <int>mjd[idx] !=
                                     current_base_mjd):
            out_mjd[out_idx] = mjd_num / denom
            out_mag[out_idx] = mag_num / denom
            out_magerr[out_idx] = sqrt(1 / denom)
            
            # Reset for the next night
            out_idx += 1
            mjd_num = 0
            mag_num = 0
            denom = 0

        if idx == num_obs:
            # Last observation, we're done.
            break
        
        current_base_mjd = <int>mjd[idx]
        inv_var = 1 / magerr[idx]**2
        mjd_num += mjd[idx] * inv_var
        mag_num += mag[idx] * inv_var
        denom += inv_var

    return out_mjd, out_mag, out_magerr
