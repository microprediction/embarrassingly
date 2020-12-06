from scipy.optimize import shgo
from embarrassingly.underpromoted import plateaudinous, Underpromoted2d
from embarrassingly.fastidious import Fastidious
import numpy as np
from pprint import pprint
import math
from microprediction import MicroReader
from embarrassingly.example_utils.kalman import kalman_error, in_sample_kalman_error, out_of_sample_kalman_error, robust_fit_out_of_sample_error
import random


# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments

# This is intended to show
#
#  (1)  Underpromotion can help a little
#  (2)  It should not be overfit ... oh the irony



mr = MicroReader()
NAMES = [ n for n in mr.get_stream_names() if not '~' in n ]
NOISE = 10


BOUNDS = [(0.1,10),(5,40)] # process noise, measurement noise


def meshit():
    name = random.choice(NAMES)
    latent = list(reversed(mr.get_lagged_values(name=name)))
    ys = [x + NOISE * np.random.randn() for x in latent]
    from embarrassingly.plot_util import mesh2d
    mesh2d(lambda x: in_sample_kalman_error(x, ys), BOUNDS, resolution=0.1)


def meta_fit(ys):
    meta_bounds = [(-2,2),(-3,0)] # log radius, log kappa
    fast_fit = Fastidious(func=robust_fit_out_of_sample_error, func_kwargs={'ys':ys, 'bounds':BOUNDS})
    meta_res = shgo(func=fast_fit, bounds=meta_bounds, n=30, iters=5, options={'minimize_every_iter': True, 'ftol': 0.001})
    print(meta_res)
    print('  Best radius '+str(math.exp(meta_res.x[0])))
    print('  Best kappa '+str(math.exp(meta_res.x[1])))
    return meta_res


if __name__=='__main__':
    x_default = np.log(np.array([1.0,0.15]))
    for _ in range(100):
        # Pick random time series and test in/out of sample errors
        name = random.choice(NAMES)
        latent = list(reversed(mr.get_lagged_values(name=name)))
        if len(latent)>900:
            ys = [x + NOISE * np.random.randn() for x in latent]
            print('For '+name)
            print('Using default radius, kappa ')
            robust_fit_out_of_sample_error(xs=x_default, verbose=True, ys=ys, bounds=BOUNDS)
            try:
                print('Searching for optimal ... ')
                meta_res = meta_fit(ys=ys)
                print('Using optimized radius, kappa ')
                robust_fit_out_of_sample_error(xs=meta_res.x, verbose=True, ys=ys, bounds=BOUNDS)
            except:
                print(' Fit failed for some reason ')
                print(' ')
        else:
            print('Too short ')
            print(' ')



