import numpy as np
from scipy.ndimage import minimum_filter1d


def setup_pyximport():
    import pyximport
    pyximport.install(reload_support=True, setup_args={'include_dirs': np.get_include()})

class cython_function():
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.function = None
        
        self.load_function()
        
    def load_function(self):
        setup_pyximport()
        self.function = getattr(__import__(self.module), self.name)
        
    def __call__(self, *args, **kwargs):
        if self.function is None:
            self.load_function()

        return self.function(*args, **kwargs)
    
    def __getstate__(self):
        # Don't return the module so that each node has to recompile it itself.
        state = self.__dict__.copy()
        state['function'] = None
        return state

    

def detect_dippers(mjd, mag, magerr, xpos, ypos, catflags, verbose=False,
                   return_mjd=False, num_sequential=3):
    '''
    a docstring
    '''
    
    # moved into here for lack of better place
    group_observations = cython_function('dipper', 'group_observations')

    if len(mjd) == 0:
        if return_mjd:
            return -1., float('nan')
        else:
            return -1.

    mjd = np.array(mjd)

    order = np.argsort(mjd)

    # Convert everything to numpy arrays and sort them by MJD
    sort_mjd = mjd[order]
    sort_mag = np.array(mag)[order]
    sort_magerr = np.array(magerr)[order]
    sort_xpos = np.array(xpos)[order]
    sort_ypos = np.array(ypos)[order]
    sort_catflags = np.array(catflags)[order]

    # Mask out bad or repeated observations.
    pad_width = 20
    x_border = 3072
    y_border = 3080

    mask = (
        (np.abs(sort_mjd - np.roll(sort_mjd, 1)) > 1e-5)
        & (sort_xpos > pad_width)
        & (sort_xpos < x_border - pad_width)
        & (sort_ypos > pad_width)
        & (sort_ypos < y_border - pad_width)
        & (sort_catflags == 0)
        
        # In the oct19 data, some observations have a magerr of 0 and aren't flagged.
        # This causes a world of problems, so throw them out.
        & (sort_magerr > 0)
        
        # In the oct19 data, a lot of dippers are the result of bad columns...
        # Unfortunately, in this version of the ZTF data we don't know which amplifier
        # everything came from. To get a reasonably clean sample (with some unnecessary
        # attrition), we cut any observations that are in the "bad" x ranges.
        & ((sort_xpos < 24) | (sort_xpos > 31))
        & ((sort_xpos < 95) | (sort_xpos > 106))
        & ((sort_xpos < 328) | (sort_xpos > 333))
        & ((sort_xpos < 1169) | (sort_xpos > 1177))
        & ((sort_xpos < 1249) | (sort_xpos > 1257))
        & ((sort_xpos < 1339) | (sort_xpos > 1349))
        & ((sort_xpos < 2076) | (sort_xpos > 2100))
        & ((sort_xpos < 2521) | (sort_xpos > 2537))
        & ((sort_xpos < 2676) | (sort_xpos > 2682))
        & ((sort_xpos < 2888) | (sort_xpos > 2895))
    )

    if np.sum(mask) < 10:
        # Require at least 10 observations to have reasonable statistics.
        if return_mjd:
            return -1., float('nan')
        else:
            return -1.
        
    mask_mjd = sort_mjd[mask]
    mask_mag = sort_mag[mask]
    mask_magerr = sort_magerr[mask]

    # Unused for now, so don't bother calculating them.
    # mask_xpos = sort_xpos[mask]
    # mask_ypos = sort_ypos[mask]
    # mask_catflags = sort_catflags[mask]
        
    use_mjd, use_mag, use_magerr = group_observations(mask_mjd, mask_mag, mask_magerr)
    
    # For well-measured observations, use the core standard deviation. For poorly
    # measured ones, use the measured standard deviation. The core standard deviation
    # should be very similar to the measured ones for stable light curves, so we
    # shouldn't be adding these in quadrature. Instead, we take whichever value is
    # larger.
    #core_std = np.std(use_mag)
    
    # NMAD
    core_std = 1.4826 * np.nanmedian(np.abs(use_mag - np.nanmedian(use_mag)))
    
    use_magerr[use_magerr < core_std] = core_std
    
    scores = (use_mag - np.median(use_mag)) / use_magerr

    # Get the minimum score for a run.
    filtered_scores = minimum_filter1d(scores, num_sequential, mode='constant')

    max_loc = np.argmax(filtered_scores)
    result = float(filtered_scores[max_loc])
    max_mjd = use_mjd[max_loc]

    if verbose:
        print("Max mjd: ", max_mjd)

    if return_mjd:
        return result, max_mjd
    else:
        return result
    
    
def detect_dippers_row(row, band='r', *args, **kwargs):
    return detect_dippers(row[f'mjd_{band}'], row[f'mag_{band}'],
                          row[f'magerr_{band}'], row[f'xpos_{band}'], row[f'ypos_{band}'],
                          row[f'catflags_{band}'], *args, **kwargs)