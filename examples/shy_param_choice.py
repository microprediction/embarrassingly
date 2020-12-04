from scipy.optimize import shgo
from embarrassingly.shy import Shy, slow_and_pointless_and_fast
from pprint import pprint
import math
from smt.surrogate_models.idw import IDW

# Run an optimizer to see which meta-parameters result in a successful yet speedy search
# We'd really want to do this across many different problems, obviously, but it illustrates tuning.


def meta_search(f, f_bounds, param_bounds ):
    """
        f            Objective
        f_bounds     Objective bounds
        param_bounds Parameter ranges
    """

    def objective_objective(xs):
        """
            Evaluate a choice of shy objective callable (varying the acceptance function and surrogate exponent)
        """
        log_t_unit = xs[0]
        log_d_unit = xs[1]
        log_eta = xs[2]
        p = xs[3]
        t_unit = math.exp(log_d_unit)
        d_unit = math.exp(log_t_unit)
        eta = math.exp(log_eta)
        surrogate_model = IDW(p=p)
        f_tilde = Shy(slow_and_pointless_and_fast, bounds=f_bounds, t_unit=t_unit, d_unit=d_unit,
                          eta=eta, surrogate_model=surrogate_model)
        f_tilde.faketime = True
        f_tilde.verbose = False
        res = shgo(func=f_tilde, bounds=f_bounds, n=10, iters=4, options={'minimize_every_iter': True, 'ftol': 0.00001})
        cpu_time = f_tilde.found_tau[-1]
        return cpu_time/100 + res.fun

    meta_res = shgo(func=objective_objective, bounds=param_bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.00001})
    return meta_res


if __name__=='__main__':
    f_bounds = [(-1 ,1) ,(-1 ,1)]
    f = slow_and_pointless_and_fast
    param_bounds = [(-8,0.0),(-8,-0),(-12,0),(1.1,2.5)]
    meta_res = meta_search(f=f, f_bounds=f_bounds, param_bounds=param_bounds )
    pprint(meta_res)
    print('t_unit = ' + str(math.exp(meta_res.x[0])))
    print('d_unit = ' + str(math.exp(meta_res.x[1])))
    print('eta    = ' + str(math.exp(meta_res.x[2])))
    print('p      = ' + str(meta_res.x[3]))


