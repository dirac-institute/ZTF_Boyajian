
def make_udf_from_annotated_function(f):
    from pyspark.sql.functions import udf
    import inspect
    
    try:
        schema = inspect.signature(f).return_annotation
        return udf(f, schema)
    except:
        raise Exception("function should be annotated with correct PySpark SQL Schema")

def plot_model(x, model, model_params, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
        
    # plot model
    min_x = np.min(x)
    max_x = np.max(x)
    _x = np.linspace(min_x, max_x, 10000)
    _y = model(_x, *model_params)    
    plt.plot(_x, _y, **kwargs)

def plot_data(x, y, yerr, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # plot data
    plt.errorbar(x, y, yerr=yerr, fmt="o", **kwargs)
    
def plot_fit_result(x, y, yerr, fit, model, with_p0=False):
    import numpy as np
    import matplotlib.pyplot as plt

    plot_data(x, y, yerr, color="C0", label="data")
    plot_model(x, model, fit['popt'], lw=3, color="C1", label="popt")
    if with_p0:
        plot_model(x, model, fit['p0'], lw=3, color="C1", ls='--', label="p0")
    plt.legend()
    
def fit_band_around_dip(df, model, band, wiggle, p0=None):
    from windowing import around_window_udf
    from spark_curve_fit import spark_curve_fit_udf

    x = df[f'mjd_{band}']
    y = df[f'mag_{band}']
    yerr = df[f'magerr_{band}']
    dip_start = df['dip.start_mjd']
    dip_end = df['dip.end_mjd']

    window_column = f"window_{band}"

    _around_dip_df = df.withColumn(
        window_column,
        around_window_udf(
            wiggle=wiggle
        )(
            x, y, yerr, dip_start, dip_end
        )
    )

    fit_x = _around_dip_df[window_column]['x']
    fit_y = _around_dip_df[window_column]['y']
    fit_yerr = _around_dip_df[window_column]['yerr']

    fit_column = f"fit_{band}"

    if p0 is None:
        _fit_df = _around_dip_df.withColumn(
            fit_column,
            spark_curve_fit_udf(
                model,
                kwargs_from_df=['sigma']
            )(
                fit_x, fit_y, fit_yerr
            )
        )
    else:
        _fit_df = _around_dip_df.withColumn(
            fit_column,
            spark_curve_fit_udf(
                model,
                kwargs_from_df=['sigma', 'p0']
            )(
                fit_x, fit_y, fit_yerr, p0
            )
        )
    
    return _fit_df

def evaluate_in_dip(df, model, band):
    from windowing import window_udf
    from spark_curve_fit import evaluate_model_error_udf
    
    x = df[f'mjd_{band}']
    y = df[f'mag_{band}']
    yerr = df[f'magerr_{band}']
    dip_start = df['dip.start_mjd']
    dip_end = df['dip.end_mjd']
    
    _in_dip_df = df.withColumn(
        f"dip_window_{band}",
        window_udf()(
            x, y, yerr, dip_start, dip_end
        )
    )
    
    _evaluate_df = _in_dip_df.withColumn(
        f"model_error_in_dip_{band}",
        evaluate_model_error_udf(model)(
            _in_dip_df[f"dip_window_{band}.x"], _in_dip_df[f"dip_window_{band}.y"], 
            _in_dip_df[f"dip_window_{band}.yerr"], _in_dip_df[f'fit_{band}.popt']
        )
    )
    
    return _evaluate_df

def evaluate_around_dip(df, model, band, wiggle):
    from windowing import around_window_udf
    from spark_curve_fit import evaluate_model_error_udf
    
    x = df[f'mjd_{band}']
    y = df[f'mag_{band}']
    yerr = df[f'magerr_{band}']
    dip_start = df['dip.start_mjd']
    dip_end = df['dip.end_mjd']
    
    _around_dip_df = df.withColumn(
        f"around_dip_window_{band}",
        around_window_udf(wiggle=wiggle)(
            x, y, yerr, dip_start, dip_end
        )
    )
    
    _evaluate_df = _around_dip_df.withColumn(
        f"model_error_around_dip_{band}",
        evaluate_model_error_udf(model)(
            _around_dip_df[f"around_dip_window_{band}.x"], _around_dip_df[f"around_dip_window_{band}.y"], 
            _around_dip_df[f"around_dip_window_{band}.yerr"], _around_dip_df[f'fit_{band}.popt']
        )
    )
    
    return _evaluate_df

def evaluate(df, model, band):
    from spark_curve_fit import evaluate_model_error_udf
    x = df[f'mjd_{band}']
    y = df[f'mag_{band}']
    yerr = df[f'magerr_{band}']
    popt = df[f'fit_{band}.popt']
        
    _evaluate_df = df.withColumn(
        f"model_error_{band}",
        evaluate_model_error_udf(model)(
            x, y, yerr, popt
        )
    )
    
    return _evaluate_df
