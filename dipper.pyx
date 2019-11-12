import numpy as np
from scipy.ndimage import minimum_filter1d

def group_observations(double[:] mjd, double[:] mag, double[:] magerr):
    """Group observations that are on the same night"""
    # Figure out how long the output array should be
    cdef int lastval = -1
    cdef int num_nights = 0
    for i in range(len(mjd)):
        if int(mjd[i]) != lastval:
            lastval = int(mjd[i])
            num_nights += 1
        
    out_mjd = np.empty(num_nights)
    out_mag = np.empty(num_nights)
    out_magerr = np.empty(num_nights)
    
    current_base_mjd = -1
    
    # Implement a weighted mean on both mag and mjd.
    out_idx = 0
    mjd_num = 0
    mag_num = 0
    denom = 0
    
    for idx in range(len(mjd) + 1):
        # Record everything if we are at the end of a night.
        if current_base_mjd > 0 and (idx == len(mjd) or int(mjd[idx]) != current_base_mjd):
            out_mjd[out_idx] = mjd_num / denom
            out_mag[out_idx] = mag_num / denom
            out_magerr[out_idx] = np.sqrt(1 / denom)
            
            # Reset for the next night
            out_idx += 1
            mjd_num = 0
            mag_num = 0
            denom = 0
            
        
        if idx == len(mjd):
            # Last observation, we're done.
            break
        
        current_base_mjd = int(mjd[idx])
        inv_var = 1 / magerr[idx]**2
        mjd_num += mjd[idx] * inv_var
        mag_num += mag[idx] * inv_var
        denom += inv_var
    
    return out_mjd, out_mag, out_magerr

def detect_dippers(mjd, filterid, psfmag, psfmagerr, xpos, ypos, catflags, verbose=False, return_mjd=False):
    if len(mjd) == 0:
        return 0.

    order = np.argsort(mjd)

    # Throw out repeated measurements.
    ordered_mjd = np.array(mjd)[order]
    order_mask = order[np.abs(ordered_mjd - np.roll(ordered_mjd, 1)) > 1e-5]
    
    # Reorder all of our arrays and convert them to numpy arrays.
    mjd = np.array(mjd).astype(np.float64)[order_mask]
    filterid = np.array(filterid)[order_mask]
    psfmag = np.array(psfmag).astype(np.float64)[order_mask]
    psfmagerr = np.array(psfmagerr).astype(np.float64)[order_mask]
    xpos = np.array(xpos)[order_mask]
    ypos = np.array(ypos)[order_mask]
    catflags = np.array(catflags)[order_mask]
    
    grouped_mjds = []
    scores = []
    
    pad_width = 20
    x_border = 3072
    y_border = 3080

    for iter_filterid in np.unique(filterid):
        cut = (
            (filterid == iter_filterid)
            & (xpos > pad_width)
            & (xpos < x_border - pad_width)
            & (ypos > pad_width)
            & (ypos < y_border - pad_width)
            & (catflags == 0)
        )

        if np.sum(cut) < 10:
            # Require at least 10 observations to have reasonable statistics.
            continue
                
        use_mjd, use_psfmag, use_psfmagerr = group_observations(mjd[cut], psfmag[cut], psfmagerr[cut])

        core_std = np.std(use_psfmag)
        filter_scores = (use_psfmag - np.median(use_psfmag)) / np.sqrt(core_std**2 + use_psfmagerr**2)
        
        grouped_mjds.append(use_mjd)
        scores.append(filter_scores)
        
    if len(grouped_mjds) == 0:
        return 0.
        
    # Reorder scores
    grouped_mjds = np.hstack(grouped_mjds)
    final_order = np.argsort(grouped_mjds)
    grouped_mjds = grouped_mjds[final_order]
    scores = np.hstack(scores)[final_order]
                
    # Check for sequential runs.
    
    # Get the minimum score for a run.
    filtered_scores = minimum_filter1d(scores, 3, mode='constant')
        
    result = float(np.max(filtered_scores))
    max_mjd = grouped_mjds[np.argmax(filtered_scores)]

    if verbose:
        print("Max mjd: ", max_mjd)

    if return_mjd:
        return result, max_mjd
    else:
        return result
