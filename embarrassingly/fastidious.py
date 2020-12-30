from deap.benchmarks import schwefel
import time
from scipy.optimize import shgo
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# See https://www.microprediction.com/blog/robust-optimization for explanation


class Fastidious:
    """ For when you want your objective function to keep a record of calls and
        running minima found. Can also save one lambda if you need func_args or func_kwargs.

        TODO: Maybe use https://pypi.org/project/persist-queue/
    """

    def __init__(self, func, func_args=None, func_kwargs=None, fake_time=False):
        """
        :param func:
        :param func_args:    Additional args as tuple
        :param func_kwargs   Additional keyword args as dict
        :param fake_time     If set to True, func should return tuple value,cpu_time
        """
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.fake_time = fake_time

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

    def _timed_call(self, x, *args, **kwargs ):
        if self.fake_time:
            f_ans = self.func(x, *args, **kwargs )
            assert len(f_ans)==2, 'As fake_time=False, the func supplied should return a 2-tuple'
            return f_ans[0], f_ans[1]
        else:
            start_time = time.time()
            y = self.func(x, *args, **kwargs)
            t = time.time() - start_time
            return y, t

    def update_minimum(self, x, y, t):
        if self.func_min is None or y < self.func_min:
            self.func_min = y
            self.log_new_minima(x=x, y=y, t=t)

    def interpret_args(self, args, kwargs):
        if (len(args) or len(kwargs)) and ((self.func_args is not None) or (self.func_kwargs is not None)):
            raise ValueError(
                'Function was created with arguments already supplied, so cannot be called with new args supplied')
        else:
            args = self.func_args or tuple()
            kwargs = self.func_kwargs or dict()
        return args, kwargs

    def _call_fastidious(self, x, *args, **kwargs):
        args, kwargs = self.interpret_args(args, kwargs)
        y, t = self._timed_call(x, *args, **kwargs)
        self.log_evaluation(x=x, y=y, t=t)
        self.update_minimum(x=x, y=y, t=t)
        return y, t

    def __call__(self, x, *args,**kwargs):
        y, t = self._call_fastidious(x=x,*args,**kwargs)
        return y

    def visualize_2d_domain(self, step=1, label=''):
        """ Visualize search pattern """
        x0s = [x[0] for x in self.train_x]
        x1s = [x[1] for x in self.train_x]
        for l in range(0, len( x0s ), step):
            plt.scatter( x0s[:l], x1s[:l], c='b' )
            plt.pause(0.001)
        x0s_found = [x[0] for x in self.found_x]
        x1s_found = [x[1] for x in self.found_x]
        plt.scatter( x0s_found, x1s_found, c='g')

        plt.title(label + ' min=' + str(self.found_y[-1]))

    def visualize_progress(self):
        plt.plot(self.found_tau, self.found_y)



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