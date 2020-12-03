from deap.benchmarks import schwefel
import time
from scipy.optimize import shgo
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class Fastidious:
    """ For when you want your objective function to keep a record of calls and
        running minima found. Can also save one lambda if you need func_args or func_kwargs.

        TODO: Maybe use https://pypi.org/project/persist-queue/
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

        # Surrogate training data ... evaluated function points only
        self.train_x = None
        self.train_y = None
        self.train_t = None

        # Log of minima found, and when
        self.func_min = None    # Running minimum func
        self.found_x = None     # Sequence of x values for new running minima
        self.found_y = None     # Sequence of minimum y values
        self.found_tau = None   # Cumulative func cpu time when minima found

    def log_evaluation(self, x, y, t):
        """ Update log of computed training data for surrogate models
        :param x:
        :param y:   value returned
        :param t:   time taken for evaluation of function (cost)
        :return:
        """
        if self.train_x is None:
            self.train_x = [np.array(x)]
            self.train_y = [np.array(y)]
            self.train_t = [np.array(t)]
        else:
            self.train_x.append(np.array(x))
            self.train_y.append(np.array(y))
            self.train_t.append(np.array(t))

    def log_new_minima(self, x, y, t):
        """ Update log of when minima are discovered.
        :param x:
        :param y:   value returned
        :param t:   function evaluation cost (time)
        :return:
        """
        if self.found_x is None:
            self.found_x = [np.array(x)]
            self.found_y = [np.array(y)]
            self.found_tau = [np.array(t)]
        else:
            self.found_x.append(np.array(x))
            self.found_y.append(np.array(y))
            self.found_tau.append(np.array(self.found_tau[-1] + t))

    def nfev(self):
        """ Total function evaluations """
        return len(self.train_x)

    def tfev(self):
        """ Total time spent evaluating function calls """
        return np.sum(self.train_t)

    def distance_to_cache(self, x, default_dist, metric='euclidean'):
        """ Distance to nearest point in cache """
        return default_dist if self.train_x is None else cdist([x], self.train_x, metric=metric)[0][0]

    def __call__(self, x):

        def timed_call(x):
            start_time = time.time()
            y = self.func(x, *self.func_args, **self.func_kwargs)
            t = time.time() - start_time
            return y, t

        y, t = timed_call(x)
        self.log_evaluation(x=x, y=y, t=t)
        if self.func_min is None or y<self.func_min:
            self.func_min = y
            self.log_new_minima(x=x, y=y, t=t)
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
        plt.plot(SAP.found_tau, SAP.found_y)
        plt.xlabel('CPU Time')
        plt.ylabel('Minimum')
        plt.yscale('log')
        label = "a="+str(round(alf, 3))
        labels.append(label)
        plt.legend(labels)
        plt.pause(0.00001)
    pass