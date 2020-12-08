from scipy.optimize import shgo
from embarrassingly.underpromoted import plateaudinous, Underpromoted2d
from embarrassingly.fastidious import Fastidious
import numpy as np
from pprint import pprint
import math
from microprediction import MicroReader
from embarrassingly.demonstrative.quirky import in_sample_quirky_error,\
    robust_fit_out_of_sample_error, robust_fit_out_of_sample_error_report
import random
from embarrassingly.demonstrative.tabulation import make_row, RESULTS_TEMPLATE_TABLE, short_name


# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments
# Fit quirky model

mr = MicroReader()
NAMES = [ n for n in mr.get_stream_names() if '~' not in n ]
NOISE = 0.00001


BOUNDS = [(-1,1), (-1,1),(-5,5),(-5,5)]

CATEGORIES = ['electricity','emoji','hospital','airport','helicopter','traffic']

if __name__=='__main__':
    x_default = np.log(np.array([0.1, 0.5]))  # radius, kappa
    rows = ''
    unique_names = set()
    NUM_UNIQUE = 25
    NUM_TOTAL = 50
    SELECTED_NAMES = [n for n in NAMES if any(s in n for s in CATEGORIES) ]
    SELECTED_NAMES = NAMES
    FORCE_UNIQUE = False
    DIFFERENCE = False
    ratios = list()
    for _ in range(20):
        # Pick random time series and test in/out of sample errors
        name = random.choice(SELECTED_NAMES)
        latent = list(reversed(mr.get_lagged_values(name=name)))
        devo = np.std(np.diff(latent))
        if devo>0.1:
            scaled = [ l/devo for l in latent ]

            if DIFFERENCE:
                scaled = np.diff(scaled)
            if len(scaled)>900 and ( not FORCE_UNIQUE or (not short_name(name) in unique_names) ):
                ys = [x + NOISE * np.random.randn() for x in scaled]
                print('For '+name)
                print('Using default radius, kappa ')
                xs = x_default
                errs = robust_fit_out_of_sample_error_report(xs=x_default, verbose=True, ys=ys, bounds=BOUNDS)
                ratios.append(errs[0][1][1]/errs[0][0][1])
                row = make_row(errs=errs[0], name=name)
                unique_names.add(short_name(name))
                rows += row
                print('Mean ratio ' + str(np.mean(ratios)))
            else:
               pass
        if len(unique_names)>=NUM_UNIQUE or len(ratios)>NUM_TOTAL:
            break
    html_report = RESULTS_TEMPLATE_TABLE.replace('ALLROWS',rows)
    print(' ')
    print(html_report)
    print(' ')
    print('Mean ratio '+str(np.mean(ratios)))



