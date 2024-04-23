import numpy as np

def flowtype_function(U__,c__):
    # ---- routine to get the flowtype --- # 
    if np.abs(U__ - c__) < 1e-10:
            type = "critical"
    elif U__ - c__ < (-1)*(1e-10):
            type = "subcritical"
    elif U__ - c__ > 1e-10:
            type = "supercritical"
    return type

if '__name__' == '__main__':
       main()