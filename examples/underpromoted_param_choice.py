from scipy.optimize import shgo
from embarrassingly.underpromoted import plateaudinous, Underpromoted2d
import numpy as np
from pprint import pprint
import math

# Trying to get a sense of what a good gain parameter might be
# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments

def helipad(x,height,radius):
    """ Includes a helicopter landing pad when you turn it upside down """
    r = np.linalg.norm(x)
    x0 = np.array([0.25,0.25])
    amp = r*math.sin(16*r*r)
    return -1-height if np.linalg.norm(x-x0)<radius else 0.1*x[0] + amp


def heli_search():
    """
       Using the helipad to search for decent parameters, especially the gain
    """

    f_bounds = [(-1, 1), (-1, 1)]
    param_bounds = [(0, 1.0), (0.001, 0.2)]
    targets = [np.array([(0.25,0.25)]),
               np.array([(0.4,0.4)]),
               np.array([(-0.1,0.15)])]
    heights = [-0.3, -0.1, 0 , 0.1, 0.3]
    radii   = [0.01, 0.02]

    def objective_objective(xs):
        """
            Evaluate a choice of objective callable by moving the helipad around and changing its size
        """
        kappa  = xs[0]
        radius = xs[1]
        running_metric = 0
        for target in targets:
            for height in heights:
                for heli_radius in radii:
                    f = lambda x: helipad(x,height=height,radius=heli_radius)
                    f_tilde = Underpromoted2d(f, bounds=f_bounds, radius=radius, kappa=kappa)
                    f_tilde.verbose = False
                    res1 = shgo(func=f_tilde, bounds=f_bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.00001})
                    distance_from_target = np.linalg.norm( np.array(res1.x)-np.array(target) )
                    cpu_time = f_tilde.found_tau[-1]
                    metric = cpu_time/1000. + distance_from_target
                    running_metric += metric
        return running_metric

    meta_res = shgo(func=objective_objective, bounds=param_bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.000001})
    return meta_res


if __name__=='__main__':

    meta_res = heli_search()
    pprint(meta_res)
    print('Kappa  = ' + str(meta_res.x[0]))
    print('Radius = '+str(meta_res.x[1]))
