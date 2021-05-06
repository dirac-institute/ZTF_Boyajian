from pyspark.sql.types import ArrayType, FloatType
from .fit_utils import make_udf_from_annotated_function

def skew_normal(x, skew, loc, xscale, yscale, offset):
    from scipy.stats import skewnorm
    _dist = skewnorm(skew, loc=loc, scale=xscale)
    return yscale * _dist.pdf(x) + offset

def skew_normal_p0(x, y, yerr, dip_start, dip_end, dip_integral) -> ArrayType(FloatType()):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    in_dip_window = (x > dip_start) & (x < dip_end)
    y_in_window  = y[in_dip_window]
    
    if len(y_in_window) > 0:
        offset = float(np.median(y))
        yscale = float(dip_integral)
        loc = float((dip_end + dip_start) / 2)
        xscale = float(dip_end - dip_start)
        skew = 0.0
    else:
#         raise RuntimeError("no measurements in window")
        offset = 1.
        yscale = 1.
        loc = 1.
        xscale = 1.
        skew = 0.

    
    return [skew, loc, xscale, yscale, offset]

skew_normal_p0_udf = make_udf_from_annotated_function(skew_normal_p0)

def top_hat(x, loc, width, depth, offset):
    import numpy as np
    x = np.array(x)
    left = loc - width / 2
    right = loc + width / 2
    outside = (x < left) | (x > right)
    inside = np.logical_not(outside)
    
    y = np.zeros(x.shape)
    y[outside] = offset
    y[inside] = offset + depth
    
    return y

def top_hat_p0(x, y, yerr, dip_start, dip_end, dip_integral) -> ArrayType(FloatType()):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    in_dip_window = (x > dip_start) & (x < dip_end)
    y_in_window  = y[in_dip_window]
    
    if len(y_in_window) > 0:
        offset = float(np.median(y))
        depth = float(dip_integral)
        loc = float((dip_end + dip_start) / 2)
        width = float(dip_end - dip_start)
    else:
        offset = 1.
        depth = 1.
        loc = 1.
        width = 1.
    
    return [loc, width, depth, offset]

top_hat_p0_udf = make_udf_from_annotated_function(top_hat_p0)