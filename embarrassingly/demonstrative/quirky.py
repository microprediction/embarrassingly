import math
import numpy as np
from scipy.optimize import shgo
from embarrassingly.underpromoted import Underpromoted2d
from pprint import pprint

# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments

# Slightly silly polynomial model

MAX_ERR = 10  # Trimmed err for normalized TS


def poly_error(xs, ys):
    """
    :param xs:   np.array parameters
    :param ys:   time series observed with noise, or list of the same
    :return:


    """

    def pred_errs(xs, ys):
        """ Model prediction error
              xs -
              ys
        """

        def pos(b):
            return max(b, 0)

        errors = list()
        for t, yi in enumerate(ys):
            if t >= 4:
                y_hat = xs[0] * ys[t - 1] + \
                        xs[1] * ys[t-2] + \
                        xs[2]*ys[t-3]*(ys[t-2]-ys[t-1])/(0.1+abs(ys[t-2]-ys[t-4])) \
                        + xs[3]*ys[t-1]*pos(ys[t-1]-ys[t-3])
                errors.append(yi - y_hat)
        return math.sqrt(np.mean(np.minimum(MAX_ERR ** 2, np.array(errors) ** 2)))

    return pred_errs(xs=xs, ys=ys)


def in_sample_quirky_error(xs, ys):
    if len(ys) > 500:
        return poly_error(xs=xs, ys=ys[:50])
    else:
        return 0.0


def out_of_sample_quirky_error(xs, ys):
    if len(ys) >= 750:
        return poly_error(xs=xs, ys=ys[500:])
    else:
        return 0.0


def robust_fit_out_of_sample_error(xs, ys, bounds, verbose=False) -> float:
    errs = robust_fit_out_of_sample_error_report(xs=xs, ys=ys, bounds=bounds, verbose=verbose)
    return float(np.mean(np.mean(errs)))


def robust_fit_out_of_sample_error_report(xs, ys, bounds, verbose=False) -> [[float]]:
    """
    :param xs:
    :param ys:    time series, or list of time series
    :param bounds:
    :param verbose:
    :return:
    """
    radius = math.exp(xs[0])
    kappa = math.exp(xs[1])

    if isinstance(ys[0], list):
        all_ys = ys
    else:
        all_ys = [ys]

    all_errs = list()
    for ys in all_ys:
        ftol = 0.00002
        n = 25
        iters = 5
        minimize_every_iter = True
        # Without underpromotion ...
        res = shgo(func=in_sample_quirky_error, bounds=bounds, args=(ys,), n=n, iters=iters,
                   options={'minimize_every_iter': minimize_every_iter, 'ftol': ftol})
        in_sample_error = in_sample_quirky_error(res.x, ys)
        out_sample_error = out_of_sample_quirky_error(res.x, ys)
        print('params:')
        print(res.x)
        # With underpromotion
        in_sample_tilde = Underpromoted2d(in_sample_quirky_error, radius=radius, kappa=kappa, bounds=bounds,
                                          func_kwargs={'ys': ys})
        in_sample_tilde.verbose = True
        res1 = shgo(func=in_sample_tilde, bounds=bounds, n=n, iters=iters,
                    options={'minimize_every_iter': minimize_every_iter, 'ftol': ftol})
        print('params with under-promotion:')
        print(res1.x)
        in_sample_error_tilde = in_sample_quirky_error(res1.x, ys)
        out_sample_error_tilde = out_of_sample_quirky_error(res1.x, ys)
        errs = [[in_sample_error, out_sample_error], [in_sample_error_tilde, out_sample_error_tilde]]
        all_errs.append(errs)
    if verbose:
        print(errs)
    return all_errs
