import numpy as np

def cylindrical_metric(x, y, a=1, phi_eval=False):
    d = np.abs(y - x)
    # vectorized verison for evaluating radial basis functions
    if phi_eval:
        return np.sqrt(d[...,:,0]**2 + d[...,:,1]**2 + np.minimum(a*d[...,:,2], a*(2*np.pi-d[...,:,2]))**2)
    # nonvectorized version of boundary point detection
    else:
        return np.sqrt(d[0]**2 + d[1]**2 + np.minimum(a*d[2], a*(2*np.pi-d[2]))**2)
