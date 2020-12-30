from scipy.optimize import shgo
from embarrassingly.underpromoted import plateaudinous, Underpromoted2d
from embarrassingly.fastidious import Fastidious
import numpy as np
from pprint import pprint
import math
from microprediction import MicroReader
from embarrassingly.demonstrative.kalman import kalman_error, in_sample_kalman_error, out_of_sample_kalman_error,\
    robust_fit_out_of_sample_error, robust_fit_out_of_sample_error_report
import random
from embarrassingly.demonstrative.tabulation import make_row, RESULTS_TEMPLATE_TABLE, short_name


# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments
# Not much to see here. Kalman is pretty stable though there is some small lift for short time series.


mr = MicroReader()
NAMES = [ n for n in mr.get_stream_names() if not '~' in n ]
NOISE = 10


BOUNDS = [(0.1,40),(1,10)] # process noise, measurement noise


def meshit():
    name = random.choice(NAMES)
    latent = list(reversed(mr.get_lagged_values(name=name)))
    ys = [x + NOISE * np.random.randn() for x in latent]
    from embarrassingly.scatterbrained import mesh2d
    mesh2d(lambda x: in_sample_kalman_error(x, ys), BOUNDS, resolution=0.1)


def meta_fit(ys):
    meta_bounds = [(-2,2),(-3,0)] # log radius, log kappa
    fast_fit = Fastidious(func=robust_fit_out_of_sample_error, func_kwargs={'ys':ys, 'bounds':BOUNDS})
    meta_res = shgo(func=fast_fit, bounds=meta_bounds, n=30, iters=5, options={'minimize_every_iter': True, 'ftol': 0.0001})
    print(meta_res)
    print('  Best radius '+str(math.exp(meta_res.x[0])))
    print('  Best kappa '+str(math.exp(meta_res.x[1])))
    return meta_res

OVERFIT = False

if __name__=='__main__':
    x_default = np.log(np.array([1.0, 0.125]))  # radius, kappa
    rows = ''
    unique_names = set()
    NUM_UNIQUE = 20
    NUM_TOTAL = 120
    SELECTED_NAMES = [n for n in NAMES if 'electricity' in n]
    FORCE_UNIQUE = False
    DIFFERENCE = False
    ratios = list()
    for row_ndx in range(140):
        # Pick random time series and test in/out of sample errors
        name = random.choice(SELECTED_NAMES)
        latent = list(reversed(mr.get_lagged_values(name=name)))
        if DIFFERENCE:
            latent = np.diff(latent)
        if len(latent)>900 and ( not FORCE_UNIQUE or (not short_name(name) in unique_names) ):
            ys = [x + NOISE * np.random.randn() for x in latent]
            print('For '+name)
            print('Using default radius, kappa ')
            xs = x_default
            errs = robust_fit_out_of_sample_error_report(xs=x_default, verbose=True, ys=ys, bounds=BOUNDS)
            ratios.append(errs[0][1][1]/errs[0][0][1])
            row = make_row(errs=errs[0], name=name)
            unique_names.add(short_name(name))
            rows += row
            print('Mean ratio ' + str(np.mean(ratios)))
            if OVERFIT:
                # Bad idea usually ...
                try:
                    print('Searching for optimal ... ')
                    meta_res = meta_fit(ys=ys)
                    print('Using optimized radius, kappa ')
                    robust_fit_out_of_sample_error(xs=meta_res.x, verbose=True, ys=ys, bounds=BOUNDS)
                except:
                    print(' Fit failed for some reason ')
                    print(' ')
        else:
           pass
        if len(unique_names)>=NUM_UNIQUE or len(ratios)>NUM_TOTAL:
            break
    html_report = RESULTS_TEMPLATE_TABLE.replace('ALLROWS',rows)
    print(' ')
    print(html_report)
    print(' ')
    print('Mean ratio '+str(np.mean(ratios)))



