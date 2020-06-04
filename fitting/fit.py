from pyspark.sql import SparkSession
import pyspark.sql.functions as sparkfunc
import matplotlib.pyplot as plt
import numpy as np
import axs
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from functools import partial
from scipy.optimize import curve_fit
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
from pyspark.sql.functions import size

def make_schema_for_function(f):
    from pyspark.sql.types import (
        FloatType, IntegerType, ArrayType, StringType,
        StructType, StructField
    )
    import inspect
    from typing import List
    FloatVector = List[float]
    IntVector = List[int]
    StrVector = List[str]
    
    params  = inspect.signature(model).parameters
    
    types = {
        p : params[p].annotation
        for p in params
    }
    
    type_mapping = {
        float : FloatType(),
        int : IntegerType(),
        str : StringType(),
        FloatVector : ArrayType(FloatType()),
        IntVector : ArrayType(IntegerType()),
        StrVector : ArrayType(StringType())
    }
    
    return StructType(
        [
            StructField(p, type_mapping[types[p]], True)
            for p in params
        ]
    )

def convert_function_to_udf(f):
    from pyspark.sql.functions import udf
    schema = make_schema_for_function(f)
    return udf(f, schema)

def check_args_match_f(f, msg, *args):
    import inspect
    sig = inspect.signature(f)
    params = sig.parameters
    if len(params) != len(args):
        raise RuntimeError(f"Function {f} takes {len(params)} arguments but {len(args)} were passed. {msg}")
        
def fit(
    x, y, yerr, model, init_params, *extra_cols, max_iter=2000
): 
    from time import time
    import inspect
    from scipy.optimize import curve_fit, OptimizeWarning

    t1 = time()
    model_arguments = list(inspect.signature(model).parameters.keys())
    model_parameters = model_arguments[1:]
    
    ret = {
        "params" : {
            key : float("nan")
            for key in model_parameters
        }
    }
    ret.update({
        key : float("nan")
        for key in [
            "chi_square",
            "chi_square_reduced",
            "runtime",
        ]
    })
    ret.update({
        "message" : "",
        "good_fit" : False,
    })
    
    if callable(init_params):
        try:
            if yerr is not None:
                check_args_match_f(init_params, "Were extra columns passed to init function?", x, y, yerr, *extra_cols)
                init_params = init_params(x, y, yerr, *extra_cols)
            else:
                check_args_match_f(init_params, "Were extra columns passed to init function?", x, y, *extra_cols)
                init_params = init_params(x, y, *extra_cols)
        except RuntimeError as e:
            ret['message'] = str(e)
            return ret
                
    ret.update({
        "params_init" : {
            f"{key}_init" : float(p0)
            for (key, p0) in zip(model_parameters, init_params)
        }
    })
    
    x = np.array(x)
    y = np.array(y)
        
    try:
        if len(x) <= len(init_params):
            raise RuntimeError("There must be more data points than free parameters in your model")

        if yerr is not None:
            yerr = np.array(yerr)
            res = curve_fit(
                model, 
                x, 
                y, 
                p0=init_params,
                sigma=yerr,
                maxfev=max_iter,
            )
        else:
            res = curve_fit(
                model, 
                x, 
                y, 
                p0=init_params,
                maxfev=max_iter,
            )
    except (ValueError, RuntimeError, OptimizeWarning, TypeError) as e:
        ret['message'] = str(e)
        return ret
    
    fit_params = res[0]
    
    ret.update({
        "params" : {
            key : float(value)
            for (key, value) in zip(model_parameters, fit_params)
        }
    })
    
    y_hat = model(x, *fit_params)
    if yerr is not None:
        chi_sq = np.sum(((y - y_hat) / yerr)**2)
    else:
        chi_sq = np.sum((y - y_hat)**2)

    # N = number of fit data points
    # d = number of fit parameters
    N, d = len(x), len(fit_params)
    if N > d:
        chi_sq_reduced = chi_sq / (N - d)
    else:
        chi_sq_reduced = float("nan")

    ret.update({
        "good_fit" : True,
        "message" : "success",
        "chi_square" : float(chi_sq),
        "chi_square_reduced" : float(chi_sq_reduced),
    })
        
    t2 = time()    
    
    ret.update({
        "runtime" : float(t2 - t1)
    })
    
    return ret

def fit_udf(model, initial_parameters, with_errors=True, schema_fields=None, infer=True):
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType, StringType, BooleanType
    from functools import partial
    import inspect
    
    _fit_schema_defaults = [
        StructField("good_fit", BooleanType(), False),
        StructField("message", StringType(), False),
        StructField("chi_square", FloatType(), False),
        StructField("chi_square_reduced", FloatType(), False),
        StructField("runtime", FloatType(), False),
    ]
    
    def infer_schema(model):
        sig = inspect.signature(model)
        params = sig.parameters
        param_names = list(params.keys())
        if len(param_names) > 1:
            param_names = param_names[1:]
        else:
            raise RuntimeError("model passed to fit_udf must have at least one parameter")

        schema = []
        
        param_schema = []
        for p in param_names:
            param_schema.append(StructField(p, params[p].annotation))
        schema.append(StructField("params", StructType(param_schema)))
        
        param_init_schema = []
        for p in param_names:
            param_init_schema.append(StructField(f"{p}_init", params[p].annotation))
        schema.append(StructField("params_init", StructType(param_init_schema)))
        
        return schema
    
    if infer:
        if schema_fields is None:
            if model is None:
                raise RuntimeError("must pass model to fit_udf to infer schema")
            else:
                schema_fields = infer_schema(model)
    else:
        if schema_fields is None:
            raise RuntimeError("must pass schema_fields to fit_udf if not inferring schema from model")
    
    schema = StructType(
        sum([_fit_schema_defaults, schema_fields], [])
    )
    
    print(schema)
    if with_errors:
        return udf(lambda x, y, yerr, *extra_cols: fit(x, y, yerr, model, initial_parameters, *extra_cols, max_iter=2000), schema)
    else:
        return udf(lambda x, y, *extra_cols : fit(x, y, None, model, initial_parameters, *extra_cols, max_iter=2000), schema)

def _fit_band(
    # light curve data
    mjd, mag, magerr,
    # dip quantification
    dip_start_mjd, dip_end_mjd,
    # model to fit
    model, init_params,
    # changes to data to fit
    dip_only=True, expand_dip=True, expansion=1.0,
):
    pass


def linear_model(x, a : FloatType(), b : FloatType()):
    return a * x + b

def init(x, y, yerr, dip_max_gap):
    a = np.mean(np.array(y) / np.array(x))
    b = np.mean(np.array(y)) - 1
    
    return [float(a), float(b)]

def model_normskew(
    x, 
    skew : FloatType(), 
    loc : FloatType(), 
    xscale : FloatType(), 
    yscale : FloatType(),
    offset : FloatType(),
):
    from scipy.stats import skewnorm
    _dist = skewnorm(skew, loc=loc, scale=xscale)
    return yscale * _dist.pdf(x) + offset

def init_normskew(x, y, yerr, dip_start, dip_end, dip_integral):
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    in_dip_window = (x > dip_start) & (x < dip_end)
    y_in_window  = y[in_dip_window]

    
    if len(y_in_window) > 0:
        offset = float(np.median(y))
        print(np.abs(y[in_dip_window] - offset))
#         yscale = float(np.max(np.abs(y[in_dip_window] - offset)))
        yscale = float(dip_integral)
        loc = float((dip_end + dip_start) / 2)
        xscale = float(dip_end - dip_start)
        skew = 0.0
    else:
        raise RuntimeError("no measurements in window")
    
    return [skew, loc, xscale, yscale, offset]
    
def around_window(x, y, yerr, start, end, wiggle=0.5) -> (
    StructType(
        [
            StructField("x", ArrayType(FloatType())),
            StructField("y", ArrayType(FloatType())),
            StructField("yerr", ArrayType(FloatType())),
        ]
    )
):
    width = end - start
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
        
    min_x = start - wiggle * width
    max_x = end + wiggle * width
    
    in_window = (x > min_x) & (x < max_x)
    
    return {
        "x" : x[in_window].tolist(),
        "y" : y[in_window].tolist(),
        "yerr" : yerr[in_window].tolist(),
    }
    
    
def make_udf_from_function(f):
    from pyspark.sql.functions import udf
    import inspect
    try:
        schema = inspect.signature(around_window).return_annotation
        return udf(f, schema)
    except:
        raise Exception("function should be annotated with correct PySpark SQL Schema")
        
        
def plot_model(lc, band, wiggle, save=False, plot_model=True):
    plt.rc("figure", figsize=(10, 8))
    plt.rc("font", size=22)

    print(lc['dip']['significance'])
    # print(lc['dip_start_mjd'], lc['dip_end_mjd'])
    # print(fits[0]['window']['x'])
    width = lc['dip']['end_mjd'] - lc['dip']['start_mjd']
    x_min = lc['dip']['start_mjd'] - width * wiggle
    x_max = lc['dip']['end_mjd'] + width * wiggle

    params = lc[f'fit_{band}']['params'].asDict()

    # params['xscale_init'] = 1

    
    x = np.array(lc[f'mjd_{band}'])
    y = np.array(lc[f'mag_{band}'])
    yerr = np.array(lc[f'magerr_{band}'])
    plt.errorbar(x, y, yerr=yerr, fmt="o", label=f"ZTF-{band}")
    plt.xlim(x_min, x_max)
    plt.axvline(lc['dip']['start_mjd'], color="k", ls="--")
    plt.axvline(lc['dip']['end_mjd'], color="k", ls="--")
    ax = plt.gca()
    ax.invert_yaxis()

    if plot_model:
        _x = np.linspace(x_min, x_max, 1000)
        y_hat = model_normskew(_x, *list(params.values()))

        plt.plot(_x, y_hat, label="model", lw=3)
        plt.text(0.05, 0.9, f"Skew = {params['skew']:.2f}", transform=ax.transAxes)
    
    plt.title(f"objid {lc['ps1_objid']}")
    plt.legend(loc="lower right")
        
    plt.xlabel("MJD")
    plt.ylabel("Magnitude")
    
    if save:
        plt.savefig(f"lcs/{lc['ps1_objid']}_{band}_band_fit.jpg")
    plt.show()
    
def fit_band(df, band, wiggle, only_good=True):
    columns = [
        "ra",
        "dec",
        "zone",
        "ps1_objid",
        f"mjd_{band}",
        f"mag_{band}",
        f"magerr_{band}",
        "dip",
    ]
    
    # windowing
    df = df.select(
        *columns, 
        make_udf_from_function(
            partial(around_window, wiggle=wiggle)
        )(
            df[f'mjd_{band}'], 
            df[f'mag_{band}'],
            df[f'magerr_{band}'],
            df['dip']['start_mjd'], 
            df['dip']['end_mjd'],
        ).alias(
            "window"
        )
    )

    columns.append("window")
    # fitting function
    df = df.select(
        *columns,
        fit_udf(
            model_normskew, 
            init_normskew,
            with_errors=True
        )(
            df['window']['x'], 
            df['window']['y'], 
            df['window']['yerr'], 
            df['dip']['start_mjd'], 
            df['dip']['end_mjd'],
            df['dip']['integral'],
        ).alias(
            f"fit_{band}"
        )
    )
    
    if only_good:
        df = df.where(
            df[f'fit_{band}']['good_fit'] == True
        )
    
    return df

def preprocess(df, bands, limit=None):
    for band in bands:
        df = df.where(
            (size(df[f'mjd_{band}']) > 0) &
            (size(df[f'mag_{band}']) > 0) &
            (size(df[f'magerr_{band}']) > 0)
        )
    
    #  sort by dip significance
    df = df.sort(
        df['dip']['significance'], ascending=False
    )
    
    bands_str = "_".join(bands)

    if limit is not None:
        df = df.limit(
            limit
        )
        limit_str = f"{limit}"

        catalog_name = f"stevengs_dippers_{bands_str}_{limit_str}"
    
        try:
            catalog.save_axs_table(df, catalog_name)
        except Exception as e:
            print(e)
            pass

        return catalog.load(catalog_name)
        
    else:
        return df