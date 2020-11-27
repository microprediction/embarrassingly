import smt
from smt.surrogate_models import IDW
import numpy as np
from tdigest import TDigest
import time
from embarrassingly.visual import mesh2d
import scipy


class Urgent:
    """ Take a function that is slow for some x and fast for others
        Turn it into a function that learns where it is fast and slow, and sometimes returns an approximate answer
        if it thinks it is about to be very slow. Occasionally it returns an actual one.

        Sometimes, this can be used to trick global optimizers into not wasting time.

    """

    def __init__(self, func, bounds, t_percentile=0.1):
        """
        :param func:
        :param bounds:
        :param t_percentile:
        """
        self.dim = len(bounds)
        self.big_distance = np.linalg.norm([b[1] - b[0] for b in bounds])
        self.func = func
        self.bounds = bounds
        self.surrogate_for_y = IDW(p=2)
        self.surrogate_for_y.options.update({'print_global': False})
        self.surrogate_for_t = IDW(p=2)
        self.surrogate_for_t.options.update({'print_global': False})
        self.cdf_t = TDigest()
        self.t_percentile = t_percentile
        self.threshold_time = 1e-4
        self.cache_x = None
        self.cache_y = None
        self.cache_t = None
        self.approx_x = None
        self.approx_y = None

    def predict_t(self, x:np.array):
        """ Returns as 1-d array """
        return self.surrogate_for_t.predict_values(np.array([x]))[0][0]

    def predict_y(self, x: np.array):
        """ Returns as 1-d array """
        return self.surrogate_for_y.predict_values(np.array([x]))[0][0]

    def distance_to_nearest_calculated(self, x):
        if self.cache_x is None:
            return self.big_distance
        else:
            return scipy.spatial.distance.cdist([x], self.cache_x)[0][0]

    def mesh_t(self):
        return mesh2d(f=self.predict_t)

    def mesh_y(self):
        return mesh2d(f=self.predict_y)

    def update_cache(self, x, y, t):
        """ Update log of computed "training" data
        :param x:
        :param y:   value returned
        :param t:   time taken
        :return:
        """
        if self.cache_x is None:
            self.cache_x = [np.array(x)]
            self.cache_y = [np.array(y)]
            self.cache_t = [np.array(t)]
        else:
            self.cache_x.append(np.array(x))
            self.cache_y.append(np.array(y))
            self.cache_t.append(np.array(t))

    def update_approx(self, x, y):
        """ Update record of approximations returned to user """
        if self.approx_x is None:
            self.approx_x = [np.array(x)]
            self.approx_y = [np.array(y)]
        else:
            self.approx_x.append(np.array(x))
            self.approx_y.append(np.array(y))

    @staticmethod
    def timed_call(f,x,*args,**kwargs):
        start_time = time.time()
        y = f(x,*args,**kwargs)
        t = time.time()-start_time
        return y,t

    def call_and_train(self, x, *args, **kwargs):
        y, t = self.timed_call(self.func,x=x,*args,**kwargs)
        self.update_cache(x=x, y=y, t=t)
        self.surrogate_for_t.set_training_values(np.array(self.cache_x), np.array(self.cache_t))
        self.surrogate_for_t.train()
        self.surrogate_for_y.set_training_values(np.array(self.cache_x), np.array(self.cache_y))
        self.surrogate_for_y.train()
        self.cdf_t.update(t)
        self.threshold_time = self.cdf_t.percentile(100 * self.t_percentile)
        return y

    def __call__(self, x, *args, **kwargs):
        x = np.asarray(x)

        if self.cache_x is None or self.distance_to_nearest_calculated(x) > self.big_distance / (2 + len(self.cache_x)):
            return self.call_and_train(x, *args, **kwargs)
        else:
            t_predicted = self.predict_t(x=x)

            if t_predicted > self.threshold_time:
                call_probability = self.threshold_time / t_predicted
            else:
                call_probability = 1
            if np.random.rand() < call_probability:
                return self.call_and_train(x=x)
            else:
                y = self.predict_y(x=x)
                self.update_approx(x=x,y=y)
                return y


def slow_and_pointless(x):
    if x[0] > 0:
        time.sleep(0.1)
    if x[1] > 0:
        time.sleep(0.3)
    return (0.25 - x[0]) ** 2 + (0.5 - x[1] * x[1])


if __name__ == '__main__':
    num_eval = 15
    rnd = np.random.rand(num_eval, 2)
    xs = rnd - 0.5 * np.ones_like(rnd)
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    SAP = Urgent(slow_and_pointless, bounds=bounds)
    for x in xs:
        SAP(x=x)
    estimate = SAP.predict_t(x=x)
    SAP.mesh_y()
    pass
