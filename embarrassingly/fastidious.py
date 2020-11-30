from deap.benchmarks import schwefel
import time
from scipy.optimize import shgo
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class Fastidious:
    """ For when you want your objective function to keep a record of calls and
        running minima found. Can also save one lambda if you need func_args or func_kwargs.
    """

    def __init__(self, func, func_args=None, func_kwargs=None):
        """
        :param func:
        :param func_args:    Additional args as tuple
        :param func_kwargs   Additional keyword args as dict
        """
        self.func = func
        self.func_args = func_args if func_args is not None else tuple()
        self.func_kwargs = func_kwargs if func_kwargs is not None else list()
        self.cache_x = None  # Record of x sent to func
        self.cache_func = None  # Record of all func(x)
        self.cache_t = None  # Time taken for the individual func call
        self.func_min = None # Running minimum func
        self.found_x = None  # Sequence of x values for new running minima
        self.found_func = None  # Sequence of minimum y values
        self.found_t = None  # Cumulative func cpu time when minima found

    def update_cache(self, x, y, t):
        """ Update log of computed "training" data
        :param x:
        :param y:   value returned
        :param t:   time taken
        :return:
        """
        if self.cache_x is None:
            self.cache_x = [np.array(x)]
            self.cache_func = [np.array(y)]
            self.cache_t = [np.array(t)]
        else:
            self.cache_x.append(np.array(x))
            self.cache_func.append(np.array(y))
            self.cache_t.append(np.array(t))

    def update_found(self, x, y, t):
        """
        :param x:
        :param y:   value returned
        :param t:   total cpu time consumed
        :return:
        """
        if self.found_x is None:
            self.found_x = [np.array(x)]
            self.found_func = [np.array(y)]
            self.found_t = [np.array(t)]
        else:
            self.found_x.append(np.array(x))
            self.found_func.append(np.array(y))
            self.found_t.append( np.array(self.found_t[-1]+t) )

    def nfev(self):
        """ Total function evaluations """
        return len(self.cache_x)

    def tfev(self):
        """ Total time spent evaluating function calls """
        return np.sum(self.cache_t)

    def distance_to_cache(self, x, default_dist, metric='euclidean'):
        """ Distance to nearest point in cache """
        return default_dist if self.cache_x is None else cdist([x], self.cache_x, metric=metric)[0][0]

    def __call__(self, x):

        def timed_call(x):
            start_time = time.time()
            y = self.func(x, *self.func_args, **self.func_kwargs)
            t = time.time() - start_time
            return y, t

        y, t = timed_call(x)
        self.update_cache(x=x, y=y, t=t)
        if self.func_min is None or y<self.func_min:
            self.func_min = y
            self.update_found(x=x,y=y,t=t)
        return y


def slow_and_painful(x,alpha):
    """ Example of a function that is sometimes slow """
    r = np.linalg.norm(x)
    quad = alpha*(0.5*0.5-r*r)/(0.5*0.5)
    compute_time = max(0,0.5*quad+x[0])
    time.sleep(compute_time)
    return schwefel([1000*x[0],980*x[1]])[0]


if __name__ == '__main__':
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    labels = list()
    for alf in np.linspace(0.05, 0.2, 5):
        SAP = Fastidious(slow_and_painful, func_kwargs={'alpha':alf})
        res = shgo(func=SAP, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
        plt.plot(SAP.found_t, SAP.found_func)
        plt.xlabel('CPU Time')
        plt.ylabel('Minimum')
        plt.yscale('log')
        label = "a="+str(round(alf, 3))
        labels.append(label)
        plt.legend(labels)
        plt.pause(0.00001)
    pass