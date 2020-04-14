import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import pyspark.sql.functions as sparkfunc
import pyspark.sql.types as stypes

from collections.abc import Iterable
from functools import partial
from collections import defaultdict
from scipy.signal import convolve

def _weighted_median(values, weights):
    """Calculate the weighted median of a set of values

    Parameters
    ----------
    values : ndarray
        List of values to perform the weighted median over.
    weights : ndarray
        List of weights to use.
    """
    order = np.argsort(values)
    sort_values = values[order]
    sort_weights = weights[order]

    percentiles = np.cumsum(sort_weights / np.sum(sort_weights))

    # Find the first element above the boundary
    loc = np.argmax(percentiles >= 0.5)

    if loc == 0:
        # First element is already above median, return it.
        return sort_values[0]
    elif percentiles[loc] == 0.5:
        # Right on a boundary, return the average of the neighboring bins.
        return (sort_values[loc] + sort_values[loc+1]) / 2.
    else:
        # In the middle of a bin, return it.
        return sort_values[loc]


def _calculate_reference_magnitude(times, mags, max_dt=1.):
    """Calculate the reference magnitude for a light curve.

    We do this by taking the median of all observations. One challenge with this is that
    if there are many repeated observations, which is the case for some of the ZTF deep
    fields, they can overwhelm all of the other observations. To mitigate this, we
    weight each observation by the time between it and the previous observation, with a
    maximum time difference specified by max_dt.

    Parameters
    ----------
    times : ndarray
        Time of each observation. This must be sorted.
    mags : ndarray
        Magnitude of each observation.
    max_dt : float
        The maximum time difference to use for weighting observations.

    Returns
    -------
    reference_magnitude : float
        The resulting reference magnitude to use.
    """
    weights = np.clip(np.diff(times, prepend=times[0] - max_dt), None, max_dt)
    reference_magnitude = _weighted_median(mags, weights)

    return reference_magnitude


def filter_ztf_observations(mjd, mag, magerr, xpos, ypos, catflags):
    """Identify and reject any bad ZTF observations, and return the valid ones.

    We reject any observations with the following characteristics:
    - Any processing flags set
    - Duplicate observations
    - Observations near the edge of a chip
    - Observations near an unflagged bad column.

    We return the resulting magnitude differences along with uncertainties, sorted by
    MJD.

    Parameters
    ----------
    mjd : list of floats
        A list of the mjd times for each observation.
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
    parsed_mjd : ndarray
        Sorted array of parsed MJDs.
    parsed_mag : ndarray
        Corresponding magnitude differences relative to the median flux
    parsed_magerr : ndarray
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

    parsed_mjd = sort_mjd[mask]
    parsed_mag = sort_mag[mask]
    parsed_magerr = sort_magerr[mask]

    return parsed_mjd, parsed_mag, parsed_magerr


def parse_light_curve(mjds, mags, magerrs, mask_window=None, min_band_observations=10):
    """Parse a light curve or set of light curves so that we can identify dips in them.

    This subtracts out a reference magnitude in each band, and returns a single light
    curve with all of the resulting measurements combined.

    Parameters
    ----------
    mjds : iterable or list of iterables
        List of observation times in each band, or the observation times in a single
        band.
    mags : iterable or list of iterables
        Corresponding measured magnitudes.
    magerrs : iterable or list of iterables
        Corresponding measured magnitude uncertainties.
    mask_window : tuple of two floats or None
        Start and end times of a window to mask out when estimating the background
        level. If None, no masking is performed.
    min_band_observations : int
        Minimum number of observations in each band. Bands with fewer than this number
        will be ignored. If there aren't enough observations, then we can't get a
        reliable estimate of the reference magnitude.

    Returns
    -------
    combined_mjd : ndarray
        Combined MJDs in each band.
    combined_mag : ndarray
        Corresponding magnitudes with the reference subtracted in each band.
    combined_magerr : ndarray
        Corresponding magnitude uncertainties in each band.
    """
    combined_mjd = []
    combined_mag = []
    combined_magerr = []

    # Check if we have a single set of observations, or if we were given a set of them
    # for each filter.
    if len(mjds) == 0:
        # No observations... return empty lists.
        return np.array([]), np.array([]), np.array([])
    elif isinstance(mjds[0], Iterable):
        # A list of observations in different filters, can use as is.
        pass
    else:
        # Observations in a single filter. Wrap them so we can treat them in the same
        # way as the previous case.
        mjds = [mjds]
        mags = [mags]
        magerrs = [magerrs]

    for mjd, mag, magerr in zip(mjds, mags, magerrs):
        mjd = np.asarray(mjd)
        mag = np.asarray(mag)
        magerr = np.asarray(magerr)

        if len(mjd) < min_band_observations:
            continue

        # Ensure that everything is sorted.
        if not np.all(np.diff(mjd) >= 0):
            # Arrays aren't sorted, fix that.
            order = np.argsort(mjd)
            mjd = mjd[order]
            mag = mag[order]
            magerr = magerr[order]

        # Mask out a window if desired.
        if mask_window is not None:
            mask = (mjd < mask_window[0]) | (mjd > mask_window[1])
            mask_mjd = mjd[mask]
            mask_mag = mag[mask]

            if len(mask_mjd) < min_band_observations:
                continue
        else:
            mask_mjd = mjd
            mask_mag = mag

        ref_mag = _calculate_reference_magnitude(mask_mjd, mask_mag)
        sub_mags = mag - ref_mag

        combined_mjd.append(mjd)
        combined_mag.append(sub_mags)
        combined_magerr.append(magerr)

    if len(combined_mjd) == 0:
        # No observations found... return empty lists.
        return np.array([]), np.array([]), np.array([])

    combined_mjd = np.hstack(combined_mjd)
    order = np.argsort(combined_mjd)
    combined_mjd = combined_mjd[order]
    combined_mag = np.hstack(combined_mag)[order]
    combined_magerr = np.hstack(combined_magerr)[order]

    return combined_mjd, combined_mag, combined_magerr


def _find_most_significant_dip_fast(mjd, mag, magerr, guess_nobs=3):
    """Find the most significant dip in a light curve.

    Parameters
    ----------
    mjd : ndarray
        Observation times. Must be sorted.
    mag : ndarray
        Corresponding measured magnitudes. The reference level must be subtracted, see
        `_calculate_reference_magnitude` and `_combine_band_light_curves` for details.
    magerr : ndarray
        Corresponding measured magnitude uncertainties.
    """
    significance = mag / magerr
    bin_edges = np.hstack([mjd[0], (mjd[1:] + mjd[:-1]) / 2., mjd[-1]])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    weighted_mag = bin_widths * mag
    weighted_var = bin_widths**2 * magerr**2
    # weighted_mag = mag
    # weighted_var = magerr**2

    old_start_idx = -1
    old_end_idx = -1

    # Guess where the dip is roughly located assuming a fixed dip length.
    conv_mag = convolve(weighted_mag, np.ones(guess_nobs), mode='same')
    conv_var = convolve(weighted_var, np.ones(guess_nobs), mode='same')
    conv_significance = conv_mag / np.sqrt(conv_var)

    start_idx = np.argmax(conv_significance)
    end_idx = np.argmax(conv_significance)

    while (old_start_idx != start_idx) | (old_end_idx != end_idx):
        old_start_idx = start_idx
        old_end_idx = end_idx

        # Update the end idx
        integrals = np.cumsum(weighted_mag[start_idx:])
        integral_vars = np.cumsum(weighted_var[start_idx:])
        significances = integrals / np.sqrt(integral_vars)
        end_idx = start_idx + np.argmax(significances)

        # Update the start idx
        integrals = np.cumsum(weighted_mag[:end_idx+1][::-1])[::-1]
        integral_vars = np.cumsum(weighted_var[:end_idx+1][::-1])[::-1]
        significances = integrals / np.sqrt(integral_vars)
        start_idx = np.argmax(significances)

        dip_significance = significances[start_idx]

    # Figure out the final properties.
    dip_start_mjd = bin_edges[start_idx]
    dip_end_mjd = bin_edges[end_idx+1]

    return dip_start_mjd, dip_end_mjd, dip_significance


def _find_most_significant_dip_fast_old(mjd, mag, magerr):
    """Find the most significant dip in a light curve.

    Parameters
    ----------
    mjd : ndarray
        Observation times. Must be sorted.
    mag : ndarray
        Corresponding measured magnitudes. The reference level must be subtracted, see
        `_calculate_reference_magnitude` and `_combine_band_light_curves` for details.
    magerr : ndarray
        Corresponding measured magnitude uncertainties.
    """
    significance = mag / magerr
    bin_edges = np.hstack([mjd[0], (mjd[1:] + mjd[:-1]) / 2., mjd[-1]])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # weighted_mag = bin_widths * mag
    # weighted_var = bin_widths**2 * magerr**2
    weighted_mag = mag
    weighted_var = magerr**2

    old_start_idx = -1
    old_end_idx = -1
    start_idx = np.argmax(significance)
    end_idx = np.argmax(significance)

    while (old_start_idx != start_idx) | (old_end_idx != end_idx):
        old_start_idx = start_idx
        old_end_idx = end_idx

        # Update the end idx
        integrals = np.cumsum(weighted_mag[start_idx:])
        integral_vars = np.cumsum(weighted_var[start_idx:])
        significances = integrals / np.sqrt(integral_vars)
        end_idx = start_idx + np.argmax(significances)

        # Update the start idx
        integrals = np.cumsum(weighted_mag[:end_idx+1][::-1])[::-1]
        integral_vars = np.cumsum(weighted_var[:end_idx+1][::-1])[::-1]
        significances = integrals / np.sqrt(integral_vars)
        start_idx = np.argmax(significances)

    dip_start_mjd = bin_edges[start_idx]
    dip_end_mjd = bin_edges[end_idx+1]
    dip_integral = np.sum(weighted_mag[start_idx:end_idx+1])
    dip_integral_uncertainty = np.sqrt(np.sum(weighted_var[start_idx:end_idx+1]))
    dip_significance = dip_integral / dip_integral_uncertainty

    result = {
        'integral': dip_integral,
        'integral_uncertainty': dip_integral_uncertainty,
        'significance': dip_significance,
        'start_mjd': dip_start_mjd,
        'center_mjd': (dip_start_mjd + dip_end_mjd) / 2.,
        'end_mjd': dip_end_mjd,
        'length': dip_end_mjd - dip_start_mjd,
    }

    return result


def _find_most_significant_dip_full(mjd, mag, magerr, min_significant_count=3,
                                    significant_threshold=3.):
    """Find the most significant dip in a light curve.

    Parameters
    ----------
    mjd : ndarray
        Observation times. Must be sorted.
    mag : ndarray
        Corresponding measured magnitudes. The reference level must be subtracted, see
        `_calculate_reference_magnitude` and `_combine_band_light_curves` for details.
    magerr : ndarray
        Corresponding measured magnitude uncertainties.
    min_significant_count : int
        The minimum number of significant observations to require in the dip.
    significant_threshold : float
        The threshold in pulls to call an observation "significant".
    """
    significance = mag / magerr
    if np.sum(significance >= significant_threshold) < min_significant_count:
        # Not enough significant observations in the light curve, can't handle this.
        return None

    bin_edges = np.hstack([mjd[0], (mjd[1:] + mjd[:-1]) / 2., mjd[-1]])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    weighted_mag = bin_widths * mag
    weighted_var = bin_widths**2 * magerr**2

    N = len(mjd)

    all_integrals = np.zeros((N, N))
    all_integral_vars = np.zeros((N, N))
    all_significant_counts = np.zeros((N, N), dtype=int)

    for i in range(N):
        if i != 0:
            all_integrals[i] = all_integrals[i-1]
            all_integral_vars[i] = all_integral_vars[i-1]
            all_significant_counts[i] = all_significant_counts[i-1]

        all_integrals[i, :i+1] += weighted_mag[i]
        all_integral_vars[i, :i+1] += weighted_var[i]
        if significance[i] >= significant_threshold:
            all_significant_counts[i, :i+1] += 1

    # Mask out entries that don't have enough significant observations.
    mask = all_significant_counts < min_significant_count
    all_integrals[mask] = 0
    all_integral_vars[mask] = 1

    # Find the most significant dip.
    all_significances = all_integrals / np.sqrt(all_integral_vars)
    end_idx, start_idx = np.unravel_index(np.argmax(all_significances),
                                          all_significances.shape)
    dip_start_mjd = bin_edges[start_idx]
    dip_end_mjd = bin_edges[end_idx+1]
    dip_integral = all_integrals[end_idx, start_idx]
    dip_integral_uncertainty = np.sqrt(all_integral_vars[end_idx, start_idx])
    dip_significance = all_significances[end_idx, start_idx]

    result = {
        'integral': dip_integral,
        'integral_uncertainty': dip_integral_uncertainty,
        'significance': dip_significance,
        'start_mjd': dip_start_mjd,
        'center_mjd': (dip_start_mjd + dip_end_mjd) / 2.,
        'end_mjd': dip_end_mjd,
        'length': dip_end_mjd - dip_start_mjd,
    }

    return result


def _get_dip_window(dip_start, dip_end, pad_fraction, min_pad_length):
    dip_length = dip_end - dip_start
    pad_length = max(min_pad_length, pad_fraction * dip_length)

    window_start = dip_start - pad_length
    window_end = dip_end + pad_length

    return window_start, window_end


def measure_dip(mjds, mags, magerrs, min_num_observations=20, min_significant_count=3,
                significant_threshold=3., window_pad_fraction=0.5,
                min_window_pad_length=5., min_significance=10.,
                return_parsed_observations=False, verbose=False, apply_cuts=False):
    """Measure the properties of a light curve, assuming that there is a single dip in
    it.

    This can handle either single single bands or multiple bands. In either case, the
    baseline level will be subtracted from each band separately.

    Parameters
    ----------
    mjds : iterable or list of iterables
        List of observation times in each band, or the observation times in a single
        band.
    mags : iterable or list of iterables
        Corresponding measured magnitudes.
    magerrs : iterable or list of iterables
        Corresponding measured magnitude uncertainties.
    """
    fail_return = defaultdict(lambda: 0)

    # Subtract the baseline level from each band, and combine their observations into a
    # single light curve.
    initial_mjd, initial_mag, initial_magerr = parse_light_curve(mjds, mags, magerrs)

    if len(initial_mjd) < min_num_observations:
        if verbose:
            print("Failed to measure dip: not enough observations.")
        return fail_return

    if apply_cuts:
        # Check if we have enough significant observations in the full light curve to
        # even bother looking for a dip.
        initial_pulls = initial_mag / initial_magerr
        total_significant_count = np.sum(initial_pulls > significant_threshold)
        if total_significant_count < min_significant_count:
            if verbose:
                print(f"Failed dip cuts: only {total_significant_count} observations "
                      f"of at least {significant_threshold} sigma, require "
                      f"{min_significant_count}.")
            return fail_return

    # Do an initial measurement of where the dip is.
    initial_dip_start, initial_dip_end, initial_dip_significance = \
        _find_most_significant_dip_fast(initial_mjd, initial_mag, initial_magerr)

    if apply_cuts and initial_dip_significance < min_significance:
        if verbose:
            print(f"Failed dip cuts: significance < {min_significance}")
        return fail_return

    # Redo the background subtraction with the dip masked out to get better background
    # estimates.
    initial_window_start, initial_window_end = _get_dip_window(
        initial_dip_start, initial_dip_end, window_pad_fraction, min_window_pad_length
    )
    mjd, mag, magerr = parse_light_curve(
        mjds, mags, magerrs, mask_window=(initial_window_start, initial_window_end)
    )

    if len(mjd) < min_num_observations:
        if verbose:
            print("Failed to measure dip: not enough observations in second pass.")
        return fail_return

    # Do the final measurement of where the dip is.
    dip_start, dip_end, dip_significance = \
        _find_most_significant_dip_fast(mjd, mag, magerr)

    if apply_cuts and dip_significance < min_significance:
        if verbose:
            print(f"Failed dip cuts: significance < {min_significance} in second pass.")
        return fail_return

    # We have a valid dip. Measure properties of it.
    result = {
        'start_mjd': dip_start,
        'end_mjd': dip_end,
        'significance': dip_significance,
    }

    # Build a mask that we can use for other measurements.
    window_start, window_end = _get_dip_window(
        dip_start, dip_end, window_pad_fraction, min_window_pad_length
    )
    window_length = window_end - window_start
    result['window_start_mjd'] = window_start
    result['window_end_mjd'] = window_end

    window_mask = (mjd > window_start) & (mjd < window_end)
    window_mjd = mjd[window_mask]
    window_mag = mag[window_mask]
    window_magerr = magerr[window_mask]
    pulls = mag / magerr
    window_pulls = pulls[window_mask]

    # Count how many significant observations are in the window and how far apart they
    # are.
    significant_mask = window_pulls >= significant_threshold
    significant_count = np.sum(significant_mask)
    significant_mjd = window_mjd[significant_mask]
    if significant_count >= 2:
        significant_length = significant_mjd[-1] - significant_mjd[0]
    else:
        significant_length = 0

    result['significant_observation_count'] = significant_count
    result['significant_length'] = significant_length

    if apply_cuts and significant_count < min_significant_count:
        # Not enough significant observations, skip the rest of the calculations.
        if verbose:
            print(f"Failed dip cuts: dip has only {significant_count} observations "
                  f"of at least {significant_threshold} sigma, require "
                  f"{min_significant_count}.")
        return fail_return

    # Integrate the dip
    bin_edges = np.hstack([mjd[0], (mjd[1:] + mjd[:-1]) / 2., mjd[-1]])
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    window_bin_widths = bin_widths[window_mask]
    result['integral'] = np.sum(window_bin_widths * window_mag)
    result['integral_uncertainty'] = np.sqrt(np.sum(window_bin_widths**2 *
                                                    window_magerr**2))

    # Find the largest gap in observations near the dip.
    # Expand out by one to get the gaps at the edges.
    gap_mask = np.convolve(window_mask, np.ones(3), mode='same') > 0
    gap_mjd = mjd[gap_mask]
    if len(gap_mjd) == 0 or gap_mask[0] or gap_mask[-1]:
        # Gap extends past the edge of observations, or there are no observations in the
        # window.
        max_gap = np.nan
        max_gap_fraction = np.nan
    else:
        max_gap = np.max(np.diff(gap_mjd))
        max_gap_fraction = max_gap / window_length
    result['max_gap'] = max_gap
    result['max_gap_fraction'] = max_gap_fraction

    # Look at the properties of residuals away from the dip.
    ref_pulls = pulls[~window_mask]
    if len(ref_pulls) > 0:
        ref_pull_std = np.std(ref_pulls)
        ref_large_pull_fraction = \
            np.sum(np.abs(ref_pulls) > significant_threshold) / len(ref_pulls)
    else:
        ref_pull_std = np.nan
        ref_large_pull_fraction = np.nan
    result['ref_observation_count'] = len(ref_pulls)
    result['ref_pull_std'] = ref_pull_std
    result['ref_large_pull_fraction'] = ref_large_pull_fraction

    if return_parsed_observations:
        # Return the parsed data used to measure the dip
        result['parsed_mjd'] = mjd
        result['parsed_mag'] = mag
        result['parsed_magerr'] = magerr

    return result


def measure_dip_ztf(mjds, mags, magerrs, all_xpos, all_ypos, all_catflags, **kwargs):
    valid_mjds = []
    valid_mags = []
    valid_magerrs = []
    for args in zip(mjds, mags, magerrs, all_xpos, all_ypos, all_catflags):
        valid_mjd, valid_mag, valid_magerr = filter_ztf_observations(*args)
        valid_mjds.append(valid_mjd)
        valid_mags.append(valid_mag)
        valid_magerrs.append(valid_magerr)

    return measure_dip(valid_mjds, valid_mags, valid_magerrs, **kwargs)


def measure_dip_row(row, *args, **kwargs):
    """Wrapper to run measure_dip_ztf on a Spark or pandas row.

    See `measure_dip_ztf` for details.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data required for `analyze_dip`
    *args
        Additional arguments to pass to `measure_dip_ztf`
    **kwargs
        Additional keyword arguments to pass to `measure_dip_ztf`

    Returns
    -------
    result : dict
        A dictionary containing the result from `measure_dip_ztf`.
    """
    result = measure_dip_ztf(
        (row['mjd_g'], row['mjd_r'], row['mjd_i']),
        (row['mag_g'], row['mag_r'], row['mag_i']),
        (row['magerr_g'], row['magerr_r'], row['magerr_i']),
        (row['xpos_g'], row['xpos_r'], row['xpos_i']),
        (row['ypos_g'], row['ypos_r'], row['ypos_i']),
        (row['catflags_g'], row['catflags_r'], row['catflags_i']),
        *args,
        **kwargs
    )

    return result


def build_measure_dip_udf(**kwargs):
    """Build a Spark UDF to run `measure_single_dip_ztf`.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to `measure_single_dip_ztf`.

    Returns
    -------
    analyze_dip_udf : function
        A wrapped function around `analyze_dip` that uses the given kwargs and that
        can be run in Spark.
    """
    use_keys = {
        'start_mjd': float,
        'end_mjd': float,
        'significance': float,
        'window_start_mjd': float,
        'window_end_mjd': float,
        'significant_observation_count': int,
        'significant_length': float,
        'integral': float,
        'integral_uncertainty': float,
        'max_gap': float,
        'max_gap_fraction': float,
        'ref_observation_count': int,
        'ref_pull_std': float,
        'ref_large_pull_fraction': float,
    }

    sparktype_map = {
        float: stypes.FloatType,
        int: stypes.IntegerType,
    }

    spark_fields = [stypes.StructField(key, sparktype_map[use_type](), True) for key,
                    use_type in use_keys.items()]
    schema = stypes.StructType(spark_fields)

    def _measure_dip_udf(
            mjd_g, mag_g, magerr_g, xpos_g, ypos_g, catflags_g,
            mjd_r, mag_r, magerr_r, xpos_r, ypos_r, catflags_r,
            mjd_i, mag_i, magerr_i, xpos_i, ypos_i, catflags_i,
            **kwargs
    ):
        result = measure_dip_ztf(
            (mjd_g, mjd_r, mjd_i),
            (mag_g, mag_r, mag_i),
            (magerr_g, magerr_r, magerr_i),
            (xpos_g, xpos_r, xpos_i),
            (ypos_g, ypos_r, ypos_i),
            (catflags_g, catflags_r, catflags_i),
            **kwargs
        )

        return [use_type(result[key]) for key, use_type in use_keys.items()]

    dip_udf = sparkfunc.udf(partial(_measure_dip_udf, **kwargs), schema)

    return dip_udf


def _plot_light_curve(row, parsed=True, show_bins=False):
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
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    band_colors = {
        'g': 'tab:green',
        'r': 'tab:red',
        'i': 'tab:purple'
    }

    for band in ['g', 'r', 'i']:
        if parsed:
            mjd, mag, magerr = filter_ztf_observations(
                row[f'mjd_{band}'],
                row[f'mag_{band}'],
                row[f'magerr_{band}'],
                row[f'xpos_{band}'],
                row[f'ypos_{band}'],
                row[f'catflags_{band}'],
            )
            if len(mjd) < 1:
                continue

            # Subtract out the zeropoint
            mjd, mag, magerr = parse_light_curve(mjd, mag, magerr)
        else:
            mask = (
                (np.array(row[f'catflags_{band}']) == 0.)
            )

            mjd = np.array(row[f'mjd_{band}'])[mask]
            mag = np.array(row[f'mag_{band}'])[mask]
            magerr = np.array(row[f'magerr_{band}'])[mask]

        ax.errorbar(mjd, mag, magerr, fmt='o', c=band_colors[band], label=f'ZTF-{band}')

    ax.set_xlabel('MJD')
    if parsed:
        ax.set_ylabel('Magnitude + offset')
    else:
        ax.set_ylabel('Magnitude')
    ax.legend()
    ax.set_title('objid %d' % row['objid'])
    ax.invert_yaxis()

    return ax

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

    ax = _plot_light_curve(row, parsed)

    if label_dip:
        start_mjd = row['dip']['start_mjd']
        end_mjd = row['dip']['end_mjd']

        ax.axvline(start_mjd, c='k', ls='--')
        ax.axvline(end_mjd, c='k', ls='--')

        if zoom:
            zoom_width = 5 * (end_mjd - start_mjd)
            ax.set_xlim(start_mjd - zoom_width, end_mjd + zoom_width)


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
                             verbose=True, both_zoom=False):
        if both_zoom or not zoom:
            plot_light_curve(rows[idx], parsed=parsed, label_dip=label_dip, zoom=False,
                             verbose=verbose)
        if both_zoom or zoom:
            plot_light_curve(rows[idx], parsed=parsed, label_dip=label_dip, zoom=True,
                             verbose=verbose)

    interact(interact_light_curve, idx=IntSlider(0, 0, max_idx))


def plot_dip(row):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    dip_result = measure_dip_row(row, return_parsed_observations=True)

    mjd = dip_result['parsed_mjd']
    mag = dip_result['parsed_mag']
    magerr = dip_result['parsed_magerr']

    ax.scatter(mjd, mag, s=5, c='k', zorder=3, label='Individual observations')

    ax.fill_between(mjd, mag, step='mid', alpha=0.2, label='Used profile')
    ax.plot(mjd, mag, drawstyle='steps-mid', alpha=0.5)

    ax.axvline(dip_result['start_mjd'], c='C2', lw=2, label='Dip boundary')
    ax.axvline(dip_result['end_mjd'], c='C2', lw=2)
    ax.axvline(dip_result['center_mjd'], c='C1', lw=2, label='Dip center')

    ax.set_xlim(dip_result['start_mjd'] - 30, dip_result['end_mjd'] + 30)
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude + offset')
