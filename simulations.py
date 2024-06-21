import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import numpy as np
import pandas as pd
import lightkurve as lk
from dipper import measure_dip

def identify_dips(res_df):
    """
    Identify dips in the results based on criteria that Kyle 
    developed.
    
    Parameters
    ----------
    res : pandas.DataFrame
        A DataFrame with the outcomes of dip searches 
        across multiple simulations 
    
    Returns
    -------
    dips : pandas.DataFrame
        A DataFrame with the results
    """
    dips = res_df[(res_df["significant_observation_count"] >= 3) &
                     (res_df["core_not_significant_fraction"] <= 0.2) &
                     (res_df["significant_width"] >= 1) &
                     (res_df["significance"] >= 5) &
                     (res_df["ref_pull_std"] < 1.5) &
                     (res_df["ref_large_pull_fraction"] < 0.1) &
                     (res_df["max_gap_fraction"] < 0.8)
                    ]
    
    return dips

def simulate_lightcurves_from_fluxes(time, flux, nbands=1, cadence=None, npoints=None, 
                         err=0.02, mean_mag=12.0):
    
    if cadence is None and npoints is None:
        raise ValueError("Either cadence or npoints need to have a value.")
    if cadence is not None:
        if cadence.shape[0] != nbands:
            raise ValueError("Need to have a set of cadences for all bands defined in `nbands`")
    if npoints is not None:
        if np.size(npoints) != nbands:
            raise ValueError("Need to have a set of `npoints` for all bands defined in `nbands`")
    if np.size(err) != 1 and len(err) != nbands:
        raise ValueError("The errors need to be either a single number or a list" + \
                         "of numbers corresponding to the number of bands.")
    if np.size(mean_mag) == 1:
        mean_mag = np.array([mean_mag])
    if np.size(err) == 1:
        err = np.array([err])
    if np.size(npoints) == 1:
        npoints = np.array([npoints])
    if cadence is not None:
        npoints = np.array([len(c) for c in cadence])
        
    lc_all = []
    # loop over different bands:
    for i in range(nbands):
        # if the cadence is not defined, randomly sample from times 
        if cadence is None:
            idx = np.sort(np.random.choice(np.arange(time.shape[0]), replace=False, size=npoints[i]))
        else:
            cad = cadence[i]
            idx = time.searchsorted(cad)

        new_time = time[idx]
        new_flux = flux[idx]
        
        mag = -2.5*np.log10(new_flux) + np.random.normal(mean_mag[i], err[i], size=npoints[i])

        magerr = np.ones_like(mag) * err[i]
            
        lc = lk.LightCurve(time=new_time, 
                                   flux=mag, 
                                   flux_err=magerr)
        lc_all.append(lc)
        
    return lc_all

def simulate_lightcurves_from_function(width, depth, tdip="random",  tseg=851, nbands=1, cadence=None, npoints=None,
                         err=0.02, mean_mag=12.0):

    if cadence is None and npoints is None:
        raise ValueError("Either cadence or npoints need to have a value.")
    if cadence is not None:
        if cadence.shape[0] != nbands:
            raise ValueError("Need to have a set of cadences for all bands defined in `nbands`")
    if npoints is not None:
        if np.size(npoints) != nbands:
            raise ValueError("Need to have a set of `npoints` for all bands defined in `nbands`")
    if np.size(err) != 1 and len(err) != nbands:
        raise ValueError("The errors need to be either a single number or a list" + \
                         "of numbers corresponding to the number of bands.")
    if np.size(mean_mag) == 1:
        mean_mag = np.array([mean_mag])
    if np.size(err) == 1:
        err = np.array([err])
    if np.size(npoints) == 1:
        npoints = np.array([npoints])
    if cadence is not None:
        npoints = np.array([len(c) for c in cadence])


    if np.size(width) == 1:
        width = np.array([width])
    if np.size(depth) == 1:
        depth = np.array([depth])

    # if tdip is None, then randomly scatter dip into the 
    # light curve, otherwise make sure it's an array
    if tdip is not None and tdip != "random":
        if np.size(tdip) == 1:
            tdip = np.array([tdip])
    elif tdip == "random":
        w = np.max(width) * 5.0
        tdip = np.random.uniform(0+w, tseg-w, size=len(width))
    else:
        if np.size(width) == 1:
            tdip = np.array([tseg/2.0])
        # if there's more than one dip, scatter through light curve
        # at regular intervals
        else:
            w = np.max(width) * 5.0
            tdip = np.linspace(w, tseg-w, len(width))

    # check that inputs are correct
    if np.any(width >= tseg):
        raise ValueError("The dip must be smaller than the length of the light curve.")

    # check that the depth is smaller than 1, or we'll have negative flux
    if np.any(depth >= 1.0):
        print(type(depth))
        print(np.any(depth) >= 1.0)
        print("depth: " + str(depth))
        raise ValueError("The depth of the dip must be < 1.")

    lc_all = []
    # loop over different bands:
    for i in range(nbands):
        # if the cadence is not defined, randomly sample from times 
        if cadence is None:
            time = np.random.uniform(0, tseg, size=npoints[i])

        else:
            time = cadence[i]

        flux = dip(time, depth, tdip, width, mean_flux=1.0)

        mag = -2.5*np.log10(flux) + np.random.normal(mean_mag[i], err[i], size=npoints[i])

        magerr = np.ones_like(mag) * err[i]

        lc = lk.LightCurve(time=time,
                           flux=mag,
                                   flux_err=magerr)
        lc_all.append(lc)

    return lc_all


def run_simulations(ztf_cadence, nsims, width, depth, tseg=851, coverage=5):
        nztf = len(ztf_cadence)
        
        #time, flux = simulate_dip_flux(tseg, coverage, width, depth)
        #tseg = time[-1] - time[0]


        lc_sims_all, res_all = [], []
        for k in range(nsims):
            idx = np.random.randint(0, nztf, size=1)

            ztf_lc = get_ztf_lightcurve(ztf_cadence, idx[0])
            max_time = np.max([ztf_lc["tseg_g"], ztf_lc["tseg_r"]])
            min_time = np.min([ztf_lc["zero_g"], ztf_lc["zero_r"]])

            #max_start = tseg - max_time

            tstart = 0.0 # np.random.uniform(0, max_start, size=1)

            c1 = ztf_lc["mjd_g"] - ztf_lc["zero_g"] + tstart
            c2 = ztf_lc["mjd_r"] - ztf_lc["zero_r"] + tstart

            cadences = np.array([c1, c2])
            nbands = 2
            mean_mag = np.array([ztf_lc["meanmag_g"], ztf_lc["meanmag_r"]])
            magerr_g = ztf_lc["magerr_g"]
            magerr_r = ztf_lc["magerr_r"]
            #magerr_g[magerr_g <= 0.0] = np.mean(magerr_g)
            #magerr_r[magerr_r <= 0.0] = np.mean(magerr_r)
            err = np.array([magerr_g, magerr_r])

            lc_all = simulate_lightcurves_from_function(width, depth, tdip="random", tseg=851, nbands=nbands, 
                                                        cadence=cadences, err=err, mean_mag=mean_mag)

            lc_sims_all.append(lc_all)

            t_all = [lc.time for lc in lc_all]
            mag_all = [lc.flux for lc in lc_all]
            magerr_all = [lc.flux_err for lc in lc_all]

            res = measure_dip(t_all, mag_all, magerr_all)
            res_all.append(res)
            
            
        res_df = pd.DataFrame(res_all)

        dips = identify_dips(res_df)
        
        return lc_sims_all, res_df, dips

def simulate_dip_flux(tseg=851, coverage=3.0, width=5, depth=0.2, tdip="random"):
    """
    Simulate normalized flux for one or multiple dips on an evenly sampled 
    grid.
    
    Parameters
    ----------
    tseg : float
        The duration of the total light curve in days
    
    coverage : float
        The number of data points per day in the finely 
        sampled light curve.
        
    width : float, or iterable
        The width of the dip in days
        
    depth : float [0,1], or iterable
        The depth of the dip, as a fraction of the flux normalized to 1. 
        Must be between 0 and 1
    
    tdip : float or None or "random"
        If no dip position is given in the time unit of the light curve, 
        either the dip is put in the middle (or equally spaced for multiple dips), if None, 
        or the position is randomized across the light curve (for "random") 
        
        
    Returns
    -------
    time : numpy.ndarray
        An array with time stamps for the flux measurements
        
    flux : numpy.ndarray
        An array with the flux measurements
    
    """
    # if my width and depth are numbers, I want to turn them 
    # into arrays so I can loop over them and won't need to 
    # treat that case separately later
    if np.size(width) == 1:
        width = np.array([width])
    if np.size(depth) == 1:
        depth = np.array([depth])
        
    # if tdip is None, then randomly scatter dip into the 
    # light curve, otherwise make sure it's an array
    if tdip is not None and tdip != "random":
        if np.size(tdip) == 1:
            tdip = np.array([tdip])
    elif tdip == "random":
        w = np.max(width) * 5.0
        tdip = np.random.uniform(0+w, tseg-w, size=len(width))
    else:
        if np.size(width) == 1:
            tdip = np.array([tseg/2.0])
        # if there's more than one dip, scatter through light curve
        # at regular intervals
        else:
            w = np.max(width) * 5.0
            tdip = np.linspace(w, tseg-w, len(width))

    # check that inputs are correct
    if np.any(width >= tseg):
        raise ValueError("The dip must be smaller than the length of the light curve.")
        
    # check that the depth is smaller than 1, or we'll have negative flux
    if np.any(depth >= 1.0):
        print(type(depth))
        print(np.any(depth) >= 1.0)
        print("depth: " + str(depth))
        raise ValueError("The depth of the dip must be < 1.")
        
    # the number of points is the days times the coverage:
    npoints = int(tseg * coverage)
    
    # create evenly sampled time array
    time = np.linspace(0.0, tseg, npoints) 

    # calculate_flux
    flux = dip(time, depth, tdip, width, mean_flux=1)
         
    return time, flux

def dip(time, depth, tdip, width, mean_flux=1):
    """
    Function to generate the flux for a dip and surrounding
    flat light curve, given a set of time stamps.

    Parameters
    ----------
    depth : float, or iterable
        A value or list of values for the depth(s) of the dip(s)

    tdip : float, or iterable
        A value or list of values for the location(s) of the dip(s)

    width : float, or iterable
        A value or list of values for the width(s) of the dip(s)

    mean_flux : float, default 1
        The constant flux of the star

    Returns
    -------
    flux : numpy.ndarray
        An array of flux values corresponding to the time stamps
        in `time`

    """
    time = np.array(time)

    # make flux
    if np.size(depth) > 1:
        flux = mean_flux
        for i in range(len(depth)):
            flux -= depth[i] * np.exp(-(time-tdip[i])**2 / (2 * width[i]**2))
    else:
        flux = mean_flux - depth * np.exp(-(time-tdip)**2 / (2 * width**2))

    return flux


def get_ztf_lightcurve(ztf_df, idx):
    """
    Get a dictionary with ZTF points in r and g bands out 
    of the cadence data frame
    
    Parameters
    ----------
    ztf_df : pd.DataFrame
        A DataFrame with the ZTF data, has columns `mjd_g`, `mag_g`, 
        `magerr_g`, `mjd_r`, `mag_r`, `magerr_r`
        
    idx: int
        An index in ztf_df.index to choose a particular light curve
        
    Returns
    -------
    ztf_lc : dict
        A dictionary with the data in a given row of the DataFrame
    """
    
    ztf_line = ztf_df.loc[idx]
    # get out data in g-band
    mjd_g = ztf_line[0]
    mag_g = ztf_line[1]
    magerr_g = ztf_line[2]
    # select only "good" observations with 
    # the magnitude error > 0
    good_g = magerr_g > 0.0
    mjd_g = mjd_g[good_g]
    mag_g = mag_g[good_g]
    magerr_g = magerr_g[good_g]
    # make sure data is sorted in time
    g_idx = np.argsort(mjd_g)
    mjd_g = mjd_g[g_idx]
    mag_g = mag_g[g_idx]
    magerr_g = magerr_g[g_idx]
  
    # get out r-band data
    mjd_r = ztf_line[3]
    mag_r = ztf_line[4]
    magerr_r = ztf_line[5]
    # pick only data points with good errors
    good_r = magerr_r > 0.0
    mjd_r = mjd_r[good_r]
    mag_r = mag_r[good_r]
    magerr_r = magerr_r[good_r]
    # sort in time
    r_idx = np.argsort(mjd_r)
    mjd_r = mjd_r[r_idx]
    mag_r = mag_r[r_idx]
    magerr_r = magerr_r[r_idx]
    
    tseg_g = mjd_g.max() - mjd_g.min()
    tseg_r = mjd_r.max() - mjd_r.min()

    ztf_lc = {"mjd_g": mjd_g, "mag_g": mag_g, "magerr_g": magerr_g,
              "mjd_r": mjd_r, "mag_r": mag_r, "magerr_r": magerr_r,
              "meanmag_g": np.mean(mag_g), "meanmag_r": np.mean(mag_r),
              "tseg_g": tseg_g, "tseg_r": tseg_r,
              "ng": len(mjd_g), "nr": len(mjd_r), 
              "zero_g": mjd_g[0], "zero_r":mjd_r[0]}
    
    return ztf_lc



