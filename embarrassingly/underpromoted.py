from embarrassingly.shy import Shy2d
from embarrassingly.scatterbrained import mesh2d
import numpy as np
import math
from scipy.optimize import shgo

# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments


class Underpromoted2d(Shy2d):

    def __init__(self, func, bounds, radius:float, kappa=0.125, **kwargs):
        """
        :param func:
        :param bounds:
        :param radius:       Measure of distance indicative of radius of plateau
        :param kappa:
        :param kwargs:
        """
        dim = len(bounds)
        d_unit = radius*math.pow(kappa,1/dim)
        shy_kwargs = {'func':func,'bounds':bounds,'t_unit':None,'d_unit':d_unit,'kappa':kappa}
        shy_kwargs.update(**kwargs)
        super().__init__(**shy_kwargs)


def plateaudinous(x):
    """ Includes a helicopter landing pad when you turn it upside down """
    r = np.linalg.norm(x)
    x0 = np.array([0.25,0.25])
    amp = r*math.sin(16*r*r)
    return -1 if np.linalg.norm(x-x0)<0.1 else 0.1*x[0] + amp


def mesh_plateaudinous():
    bounds = [(-1, 1), (-1, 1)]
    mesh2d(f, bounds)


if __name__=='__main__':
    bounds = [(-1,1),(-1,1)]
    f = plateaudinous
    res1 = shgo(func=f, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print("Global min occurs at "+str(res1.x))

    f_tilde = Underpromoted2d(f, bounds=bounds, radius=0.01)
    res2 = shgo(func=f_tilde, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Helicopter lands at '+str(res2.x))