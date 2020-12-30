import numpy as np
import time
import math
from embarrassingly.memorable import np_cache

# Example of a backtest-like objective function
# That is to say that it is the accumulation of emissions from a state machine


@np_cache(maxsize=10000)
def accumulated(xs: np.array, k: int):
    """ State machine accumulated value and state after period k """

    def step(xs: np.array, k: int, s=None):
        """ Returns increment to the objective function at k'th step
            s: initial state
            xs: parameters
            returns:  y, s'     (value increment, new state)
        """
        if s is None:
            s = 1
        time.sleep(0.03)
        return 1-(0.333333-xs[0])**2, 2 * s

    if k == 0:
        return step(xs, k=0, s=None)
    else:
        v, s_prime = accumulated(xs, k=k - 1)
        dv, s_post = step(xs, k=k, s=s_prime)
        return v + dv, s_post


def backtest_objective(xs):
    """ Smoothed version of period by period memoized accumulated results """
    # Must be digitized in space
    num_periods = 100
    c = xs[0]        # progress in (0,1)
    xs_ = np.round(xs[1:],decimals=2)
    k1 = int(math.floor(c*num_periods))
    v1, s1 = accumulated(xs_, k=k1)
    k2 = k1+1
    v2, s2 = accumulated(xs_, k=k2)
    r = (c*num_periods-k1)
    return -( v1*(1-r)+v2*r )


if __name__=='__main__':
    start_time = time.time()
    xs1 = np.array([0.025,0.5])   # 3 periods
    xs2 = np.array([0.05,0.5])   # 5 periods

    y1 = backtest_objective(xs1)
    print('elapsed = '+str(time.time()-start_time))
    y2 = backtest_objective(xs2)
    print('elapsed = '+str(time.time()-start_time))

    cs = np.linspace(0,1,1000)
    ys = [ backtest_objective( np.array([c,3.5])) for c in cs]
    import matplotlib.pyplot as plt
    plt.plot(cs,ys)
    pass

    # Mesh
    from embarrassingly.scatterbrained import mesh2d
    bounds=[(0,1.0),(-1,1)]
    mesh2d(f=backtest_objective,bounds=bounds,resolution=0.05)
    pass