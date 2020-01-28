import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import pyspark.sql.functions as sparkfunc
import pyspark.sql.types as pyspark_types

from functools import partial


def get_separated_times_mask(times, min_dt=0.5):
    """Get a mask that only selects observations that are separated by a minimum
    amount of time

    Parameters
    ----------
    times : np.array
        The list of times to consider. This must be sorted.
    min_dt : float
        The minimum time gap to allow.

    Returns
    -------
    min_time_mask : np.array
        The resulting mask, True for valid observations and False for ones that come too
        close to a previous one.
    """
    min_time_mask = (times - np.roll(times, 1)) > min_dt
    if len(min_time_mask) > 0:
        min_time_mask[0] = True
    return min_time_mask


def parse_observations(mjd, mag, magerr, xpos, ypos, catflags):
    """Parse a list of observations.

    This method first identifies the set of valid observations and rejects any that are
    suspected to be bad. We reject any observations with the following characteristics:
    - Any processing flags set
    - Observations near the edge of a chip
    - Observations near an unflagged bad column.

    We then calculate the median brightness for each light curve, and convert all
    observations to a difference relative to this median brightness. We return the
    resulting magnitude differences along with uncertainties, sorted by MJD.

    Parameters
    ----------
    mjd : list of floats
        A list of the mjd times for each observaion.
    mag : list of floats
        A list of the observed magnitudes.
    magerr : list of floats
        A list of the observed magnitude uncertanties.
    xpos : list of floats
        A list of the x positions on the CCD for each observation.
    ypos : list of floats
        A list of the y positions on the CCD for each observation.
    catflags : list of floats
        A list of the processing flags for each observation.
    
    Returns
    -------
    parsed_mjd : numpy.array
        Sorted array of parsed MJDs.
    parsed_mag : numpy.array
        Corresponding magnitude differences relative to the median flux
    parsed_magerr : numpy.array
        Magnitude uncertainties, including contributions from the intrinsic dispersion
        if applicable.
    """
    if len(mjd) == 0:
        return [], [], []

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
        
        # In the oct19 data, a lot of dips are the result of bad columns...
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
        return [], [], []
        
    mask_mjd = sort_mjd[mask]
    mask_mag = sort_mag[mask]
    mask_magerr = sort_magerr[mask]

    # Calculate statistics on the light curve. To avoid being affected by periods with
    # many rapid observations, we only consider observations that are separated by
    # a given amount of time.
    min_time_mask = get_separated_times_mask(mask_mjd)

    use_mag = mask_mag[min_time_mask]

    # Subtract the reference flux from the observations.
    base_mag = np.median(use_mag)
    parsed_mag = mask_mag - base_mag

    return mask_mjd, parsed_mag, mask_magerr


def analyze_dip(mjd_g, mag_g, magerr_g, xpos_g, ypos_g, catflags_g, mjd_r, mag_r,
                   magerr_r, xpos_r, ypos_r, catflags_r, max_gap=2.,
                   min_num_observations=3, min_dip_time=2., threshold=3.):
    """Analyze a light curve to identify the largest dip and calculate statistics on it.

    A dip is defined as all observations with decrease in the observed magnitude with a
    significance greater than the given threshold. We require that observations of the
    dip have a cadence of at most max_gap days, and that there are observations below
    the theshold on either side of the dip to ensure that we capture the full profile.
    Note that we merge the filters (after estimating a reference brightness in each
    band), and compute all of the statistics on the combined light curve.

    We choose the "largest dip" to be the one with the largest integrated magnitude.
    Dips that aren't fully resolved are rejected.

    Parameters
    ----------
    mjd_g, mag_g, ..., xpos_r, ypos_r : lists of floats
        Properties of observations in each of the g and r bands. See parse_observations
        for details.
    max_gap : float
        Maximum allowed gap between observations in a dip in days.
    min_num_observations : int
        Minimum number of observations required to be in a dip.
    min_dip_time : float
        Minimum length of a dip between the first and last observation in days.
    threshold : float
        Threshold to use when determining if an observation is a significant dip
        relative to the baseline or not, in number of standard deviations.

    Returns
    -------
    intmag : float
        Integrated magnitude for the largest dip, from the first to the last significant
        observation.
    start_mjd : float
        MJD corresponding to the first significant observation for the largest dip.
    end_mjd : float
        MJD corresponding to the last significant observation for the largest dip.
    num_observations : float
        Number of observations in the largest dip.
    complexity : float
        An attempt at measuring how complex the largest dip is. This doesn't really
        work, but should be 1 for a simple dip like a single eclipse, and a larger
        number for more complex dips.
    significance : float
        The significance of th largest dip, measured as the integrated magnitude of the
        dip divided by the NMAD (a robust estimate of the standard deviation) of the
        observations away from the dip.
    num_dips : int
        The number of different dips that were identified in the light curve.
    """
    
    parsed_mjd_g, parsed_mag_g, parsed_magerr_g = parse_observations(
        mjd_g, mag_g, magerr_g, xpos_g, ypos_g, catflags_g
    )
    parsed_mjd_r, parsed_mag_r, parsed_magerr_r = parse_observations(
        mjd_r, mag_r, magerr_r, xpos_r, ypos_r, catflags_r
    )
    
    mjd = np.hstack([parsed_mjd_g, parsed_mjd_r])
    order = np.argsort(mjd)
    mjd = mjd[order]
    mag = np.hstack([parsed_mag_g, parsed_mag_r])[order]
    magerr = np.hstack([parsed_magerr_g, parsed_magerr_r])[order]
    
    significance = mag / magerr
    
    dip_start_mjd = None
    
    best_intmag = -1.
    best_start_mjd = float('nan')
    best_end_mjd = float('nan')
    best_num_observations = 0
    best_complexity = float('nan')

    num_dips = 0

    for idx in range(1, len(mjd)):        
        if mjd[idx] - mjd[idx-1] > max_gap:
            # We have a gap in observations larger than our desired threshold.
            # The previous dip (if there was one) can't be used.
            
            # Reset
            dip_start_mjd = None
        elif significance[idx] >= threshold:
            # Found a significant observation. Increase the current nobs.

            if dip_start_mjd is None:
                if significance[idx-1] >= threshold:
                    # Continuation of a dip that we didn't identify the start of.
                    # Ignore it.
                    pass
                else:
                    # Found the start of a dip. Start recording it.
                    dip_start_mjd = mjd[idx]

                    dip_intmag = 0.
                    dip_num_observations = 1
                    dip_max_mag = mag[idx]
                    dip_sum_deltas = mag[idx] - magerr[idx]
            else:
                # Inside of a dip.
                dip_num_observations += 1
                
                # Integrate the magnitude using the trapezoid rule.
                mean_mag = (mag[idx] + mag[idx-1]) / 2.
                dt = mjd[idx] - mjd[idx-1]
                dip_intmag += dt * mean_mag
                
                if mag[idx] > dip_max_mag:
                    dip_max_mag = mag[idx]
                
                dip_sum_deltas += np.abs(mag[idx] - mag[idx-1]) - magerr[idx]
        elif dip_start_mjd is not None:
            # We found the end of a dip. Record it if it is the best one.
            dip_sum_deltas += mag[idx-1] - magerr[idx]
            
            dip_complexity = dip_sum_deltas / dip_max_mag / 2.
            
            if (dip_intmag > best_intmag
                    and dip_num_observations >= min_num_observations
                    and (mjd[idx-1] - dip_start_mjd) > min_dip_time):
                best_intmag = dip_intmag
                best_start_mjd = dip_start_mjd
                best_end_mjd = mjd[idx-1]
                best_num_observations = dip_num_observations
                best_complexity = dip_complexity

            # Reset
            dip_start_mjd = None
    
        # Count the total number of dips. We don't care if we capture the edges properly
        # for this, we just care about finding every time that we transition above the
        # threshold. This is helpful for vetoing highly variable objects.
        if significance[idx] > threshold and significance[idx - 1] < threshold:
            num_dips += 1
            
    # Get a measure of the significance of the dip by comparing the integrated size
    # of the dip to the typical variation scale of the light curve. We calculate the typical
    # variance using observations outside of the 
    mask = (
        ((mjd < best_start_mjd - 5) | (mjd > best_end_mjd + 5))
        & get_separated_times_mask(mjd)
    )

    if np.sum(mask) < 5:
        dip_significance = 0.
    else:
        mask_std = np.std(mag[mask])
        dip_significance = best_intmag / mask_std
                        
    return (
        float(best_intmag),
        float(best_start_mjd),
        float(best_end_mjd),
        int(best_num_observations),
        float(best_complexity),
        float(dip_significance),
        int(num_dips)
    )


def analyze_dip_row(row, *args, **kwargs):
    """Wrapper to run analyze_dip on a Spark or pandas row.

    See `analyze_dip` for details.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data required for `analyze_dip`
    *args
        Additional arguments to pass to `analyze_dip`
    **kwargs
        Additional keyword arguments to pass to `analyze_dip`

    Returns
    -------
    result : dict
        A dictionary containing the result from `analyze_dip`.
    """
    result = analyze_dip(
        row['mjd_g'],
        row['mag_g'],
        row['magerr_g'],
        row['xpos_g'],
        row['ypos_g'],
        row['catflags_g'],
        row['mjd_r'],
        row['mag_r'],
        row['magerr_r'],
        row['xpos_r'],
        row['ypos_r'],
        row['catflags_r'],
        *args,
        **kwargs
    )
    
    return {
        'intmag': result[0],
        'start_mjd': result[1],
        'end_mjd': result[2],
        'nobs': result[3],
        'complexity': result[4],
        'significance': result[5],
        'num_dips': result[6],
    }


def build_analyze_dip_udf(**kwargs):
    """Build a Spark UDF to run `analyze_dip`.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to `analyze_dip`.

    Returns
    -------
    analyze_dip_udf : function
        A wrapped function around `analyze_dip` that uses the given kwargs and that
        can be run in Spark.
    """

    schema = pyspark_types.StructType([
        pyspark_types.StructField("intmag", pyspark_types.FloatType(), False),
        pyspark_types.StructField("start_mjd", pyspark_types.FloatType(), True),
        pyspark_types.StructField("end_mjd", pyspark_types.FloatType(), True),
        pyspark_types.StructField("nobs", pyspark_types.IntegerType(), True),
        pyspark_types.StructField("complexity", pyspark_types.FloatType(), True),
        pyspark_types.StructField("significance", pyspark_types.FloatType(), False),
        pyspark_types.StructField("num_dips", pyspark_types.IntegerType(), False),
    ])

    func = partial(analyze_dip, **kwargs)

    analyze_dip_udf = sparkfunc.udf(func, schema)

    return analyze_dip_udf

def _plot_light_curve(row, parsed=True):
    """Helper for `plot_light_curve` to do the actual work of plotting a light curve.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    parsed : bool
        If True, the observations in each band will be passed through
        `parse_observations` before plotting them which rejects bad observations and
        subtracts the median magnitude from each filter. Otherwise, the raw observations
        are plotted.
    """
    plt.figure(figsize=(8, 6), dpi=100)

    band_colors = {
        'g': 'tab:green',
        'r': 'tab:red',
        'i': 'tab:purple'
    }

    for band in ['g', 'r', 'i']:
        if parsed:
            mjd, mag, magerr = parse_observations(
                row[f'mjd_{band}'],
                row[f'mag_{band}'],
                row[f'magerr_{band}'],
                row[f'xpos_{band}'],
                row[f'ypos_{band}'],
                row[f'catflags_{band}'],
            )
        else:
            mask = (
                (np.array(row[f'catflags_{band}']) == 0.)
            )

            mjd = np.array(row[f'mjd_{band}'])[mask]
            mag = np.array(row[f'mag_{band}'])[mask]
            magerr = np.array(row[f'magerr_{band}'])[mask]

        plt.errorbar(mjd, mag, magerr, fmt='o', c=band_colors[band], label=f'ZTF-{band}')

    plt.xlabel('MJD')
    if parsed:
        plt.ylabel('Magnitude + offset')
    else:
        plt.ylabel('Magnitude')
    plt.legend()
    plt.title('objid %d' % row['objid'])
    plt.gca().invert_yaxis()
    plt.show()


def _print_light_curve_info(row):
    """Plot information about a light curve.

    This outputs a simbad link 

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    """
    display(HTML("<a href='http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=%.6f%+.6f&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=20&Radius.unit=arcsec&submit=submit+query&CoordList='>SIMBAD</link>" % (row['ra'], row['dec'])))
    print("RA+Dec: %.6f%+.6f" % (row['ra'], row['dec']))
    

def plot_light_curve(row, parsed=True, label_dip=True, zoom=False, verbose=True):
    """Plot a light curve.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    parsed : bool
        If True, the observations in each band will be passed through
        `parse_observations` before plotting them which rejects bad observations and
        subtracts the median magnitude from each filter. Otherwise, the raw observations
        are plotted.
    label_dip : bool
        If True, the dip is labeled on the plot with vertical lines. This is ignored if
        the dip statistics haven't been calculated yet.
    zoom : bool
        If True, the plot is zoomed in around the dip.
    verbose : bool
        If True, print out information about the dip.
    """
    if 'dip' not in row:
        # Can only label the dip if it has been identified.
        label_dip = False

    if verbose:
        _print_light_curve_info(row)

        if label_dip:
            print("")
            print("Dip details:")
            for key, value in row['dip'].asDict().items():
                print(f"{key:11s}: {value}")

    _plot_light_curve(row, parsed)

    if label_dip:
        start_mjd = row['dip']['start_mjd']
        end_mjd = row['dip']['end_mjd']
        
        plt.axvline(start_mjd, c='k', ls='--')
        plt.axvline(end_mjd, c='k', ls='--')
        
        if zoom:
            plt.xlim(start_mjd - 10, end_mjd + 10)


def plot_interactive(rows):
    """Generate an interactive plot for a set of rows.

    Parameters
    ----------
    rows : List of spark rows
        A list of spark rows where each row contains all of the observation data for a
        light curve.
    """
    from ipywidgets import interact, IntSlider

    max_idx = len(rows) - 1

    def interact_light_curve(idx, parsed=True, label_dip=True, zoom=False,
                             verbose=True):
        return plot_light_curve(rows[idx], parsed=parsed, label_dip=label_dip,
                                zoom=zoom, verbose=verbose)

    interact(interact_light_curve, idx=IntSlider(0, 0, max_idx))
