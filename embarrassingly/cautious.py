from embarrassingly.shy import Shy
from embarrassingly.visual import mesh2d
import numpy as np
import math
from scipy.optimize import shgo


class Cautious(Shy):

    def __init__(self, func, bounds, radius:float, kappa=0.25, **kwargs):
        """
        :param func:
        :param bounds:
        :param radius:       Measure of distance indicative of radius of plateau
        :param kappa:
        :param kwargs:
        """
        dim = len(bounds)
        d_unit = radius*math.pow(kappa,1/dim)      # Can do better
        shy_kwargs = {'func':func,'bounds':bounds,'t_unit':1e6,'d_unit':d_unit,'kappa':kappa}
        shy_kwargs.update(**kwargs)      # User can override d_unit here if needed
        super().__init__(**shy_kwargs)


def plateaudinous(x):
    """ A helicopter landing pad when you turn it upside down """
    r = np.linalg.norm(x)
    x0 = np.array([0.25,0.25])
    amp = r*math.sin(16*r*r)
    return -1 if np.linalg.norm(x-x0)<0.1 else 0.1*x[0] + amp


def mesh_plateaudinous():
    bounds = [(-1, 1), (-1, 1)]
    mesh2d(plateaudinous, bounds)


if __name__=='__main__':
    bounds = [(-1,1),(-1,1)]
    res1 = shgo(func=plateaudinous, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    global_min_at = res1.x

    f = Cautious(plateaudinous, bounds=bounds, radius=0.01)
    res2 = shgo(func=f, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    helicopter_lands_at = res2.x