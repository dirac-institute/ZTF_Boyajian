from pyspark.sql.types import *
import numpy as np

def spark_curve_fit(
    f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, 
    bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs
) -> StructType(
       [
           StructField(
               "info", 
               StructType(
                   [
                       StructField("message", StringType()),
                       StructField("good", BooleanType()),
                       StructField("runtime", FloatType()),
                   ]
               )
           ),
           StructField(
               "popt", 
               ArrayType(FloatType())
           ),
           StructField(
               "pcov", 
               ArrayType(ArrayType(FloatType()))
           ),
           StructField(
               "p0",
               ArrayType(FloatType()),
               True
           )
       ]
       
):
    from scipy.optimize import curve_fit, OptimizeWarning
    from time import time

    t1 = time()
    
    ret = {
        "info" : {
            "message" : None,
            "runtime" : None,
            "good" : None,
        },
        "popt" : None,
        "pcov" : None,
    }
    ret['p0'] = p0
    
    try:
        popt, pcov = curve_fit(
            f, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, 
            check_finite=check_finite, bounds=bounds, method=method, jac=jac, **kwargs
        )
        popt = popt.astype(np.float64).tolist()
        pcov = pcov.astype(np.float64).tolist()
        
        ret["popt"] = popt
        ret["pcov"] = pcov
        
        ret['info']['message'] = "OK"
        ret['info']['good'] = True
    except (ValueError, RuntimeError, OptimizeWarning, TypeError) as e:
        ret['info']['message'] = str(e)
        ret['info']['good'] = False
        
    t2 = time()
    ret['info']['runtime'] = t2 - t1
    
    return ret

def spark_curve_fit_udf(f, kwargs_from_df=[], **udf_kwargs):
    import inspect
    from pyspark.sql.functions import udf

    sig = inspect.signature(spark_curve_fit)
    schema = sig.return_annotation
    
    if len(kwargs_from_df) == 0:
        _func = lambda *columns : spark_curve_fit(f, columns[0], columns[1], **udf_kwargs)
    else:
        def _with_kwargs(*columns):
            kwargs = { arg_name : col for arg_name, col in zip(kwargs_from_df, columns[2:]) }
            kwargs.update(udf_kwargs)
            return spark_curve_fit(f, columns[0], columns[1], **kwargs)
        _func = _with_kwargs
        
    return udf(
        _func, 
        schema
    )

from pyspark.sql.types import *
def evaluate_model_error(model, x, y, yerr, params) -> StructType(
    [
        StructField("sum_square_error", FloatType()),
        StructField("reduced_sum_square_error", FloatType()),
    ]
):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    N = x.shape[0]
    p = len(params)
    
    sum_square_error = np.sum(
        ((y - model(x, *params)) / yerr)**2
    )
    reduced_sum_square_error = sum_square_error / (N - p)
    
    return {
        "sum_square_error" : float(sum_square_error),
        "reduced_sum_square_error" : float(reduced_sum_square_error)
    }
    
def evaluate_model_error_udf(model):
    from fit_utils import make_udf_from_annotated_function
    from functools import partial
    
    return make_udf_from_annotated_function(partial(evaluate_model_error, model))