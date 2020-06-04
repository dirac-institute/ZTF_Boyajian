from pyspark.sql.types import *

def window(x, y, yerr, start, end) -> (
    StructType(
        [
            StructField("x", ArrayType(FloatType())),
            StructField("y", ArrayType(FloatType())),
            StructField("yerr", ArrayType(FloatType())),
        ]
    )
):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    in_window = (x > start) & (x < end)
    
    return {
        "x" : x[in_window].tolist(),
        "y" : y[in_window].tolist(),
        "yerr" : yerr[in_window].tolist(),
    }

def window_udf():
    from fit_utils import make_udf_from_annotated_function
    return make_udf_from_annotated_function(window)

def around_window(x, y, yerr, start, end, wiggle=0.5) -> (
    StructType(
        [
            StructField("x", ArrayType(FloatType())),
            StructField("y", ArrayType(FloatType())),
            StructField("yerr", ArrayType(FloatType())),
        ]
    )
):
    import numpy as np

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

def around_window_udf(**kwargs):
    from functools import partial
    from fit_utils import make_udf_from_annotated_function

    return make_udf_from_annotated_function(partial(around_window, **kwargs))
