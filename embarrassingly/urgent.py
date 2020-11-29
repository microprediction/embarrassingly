import smt
from smt.surrogate_models import IDW
import numpy as np
from tdigest import TDigest
import time
from embarrassingly.visual import mesh2d
import scipy
import math
from deap.benchmarks import h1, schwefel
import json
import matplotlib.pyplot as plt

class Urgent:
    """ Take a function that is slow for some x and fast for others
        Turn it into a function that learns where it is fast and slow, and sometimes returns an approximate answer
        if it thinks it is about to be very slow. Occasionally it returns an actual one.

        Sometimes, this can be used to trick global optimizers into not wasting time.

    """

    def __init__(self, func, bounds, d_unit, t_unit):
        """
        :param func:
        :param bounds:
        :param t_unit:     cpu time units
        """
        self.dim = len(bounds)
        self.big_distance = np.linalg.norm([b[1] - b[0] for b in bounds])
        self.func = func
        self.bounds = bounds
        self.surrogate_for_y = IDW(p=2)
        self.surrogate_for_y.options.update({'print_global': False})
        self.surrogate_for_t = IDW(p=2)
        self.surrogate_for_t.options.update({'print_global': False})
        self.t_unit = t_unit
        self.d_unit = d_unit
        self.cache_x = None
        self.cache_y = None
        self.cache_t = None
        self.approx_x = None
        self.approx_y = None
        self.approx_message = None
        self.found_x = None
        self.found_y = None
        self.found_t = None

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
        """ Show approximate and actual CPU times """
        fig, ax = mesh2d(f=self.predict_t, bounds=self.bounds)
        x1 = [x[0] for x in self.cache_x]
        x2 = [x[1] for x in self.cache_x]
        fs = [y for y in self.cache_t]
        ax.scatter(xs=x1, ys=x2, zs=fs, c='blue')
        if self.approx_x is not None:
            x1 = [x[0] for x in self.approx_x]
            x2 = [x[1] for x in self.approx_x]
            fs = [y for y in self.approx_t]
            ax.scatter(xs=x1, ys=x2, zs=fs, c='green')
            fig.show()
        return fig, ax

    def mesh_y(self):
        """ Show approximate and actual function valuations """
        fig, ax = mesh2d(f=self.predict_y, bounds=self.bounds)
        x1 = [x[0] for x in self.cache_x]
        x2 = [x[1] for x in self.cache_x]
        fs = [y for y in self.cache_y]
        ax.scatter(xs=x1, ys=x2, zs=fs)
        try:
            x1 = [x[0] for x in self.approx_x]
            x2 = [x[1] for x in self.approx_x]
            fs = [y for y in self.approx_y]
            ax.scatter(xs=x1, ys=x2, zs=fs, c='green')
            fig.show()
        except TypeError:
            pass
        return fig, ax

    def update_found(self, x, y, t):
        """
        :param x:
        :param y:   value returned
        :param t:   total cpu time consumed
        :return:
        """
        if self.found_x is None:
            self.found_x = [np.array(x)]
            self.found_y = [np.array(y)]
            self.found_t = [np.array(t)]
        else:
            self.found_x.append(np.array(x))
            self.found_y.append(np.array(y))
            self.found_t.append(np.array(t))

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

    def update_approx(self, x, y, t, message = ''):
        """ Update record of approximations returned to user """
        if self.approx_x is None:
            self.approx_x = [np.array(x)]
            self.approx_y = [np.array(y)]
            self.approx_t = [np.array(t)]
            self.approx_message = [message]
        else:
            self.approx_x.append(np.array(x))
            self.approx_y.append(np.array(y))
            self.approx_t.append(np.array(t))
            self.approx_message.append(message)

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
        if self.found_y is None or y < self.found_y[-1]:
            total_cpu = np.sum(self.cache_t)
            self.update_found(x=x,y=y,t=total_cpu)
            print('New min '+str(y)+' at '+str(x))
        return y

    def skip(self, d_ratio, t_ratio):
        """ Decide whether to skip """
        rn1 = np.random.rand()
        rn2 = np.random.rand()
        return rn1 < 1/(1e-6+d_ratio) and rn2 > 1 / (1e-6+t_ratio)

    def __call__(self, x, *args, **kwargs):
        x = np.asarray(x)
        if self.cache_x is not None:
            d = self.distance_to_nearest_calculated(x)
            t_hat = self.predict_t(x=x)
            t_ratio = t_hat/self.t_unit
            d_ratio = d/self.d_unit
            if self.skip(d_ratio=d_ratio, t_ratio=t_ratio):
                y_hat = self.predict_y(x=x)
                self.update_approx(x=x, y=y_hat, t=t_hat,
                                   message=json.dumps({'yHat': y_hat, 't_ratio':t_ratio,'distance':d }))
                #print('               Skipped '+str(t_hat)+' min = '+str(self.minimum))
                return y_hat
            else:
                return self.call_and_train(x,*args,**kwargs)
        else:
            return self.call_and_train(x,*args,**kwargs)


def slow_and_pointless(x):
    """ Example of a function that is sometimes slow """
    r = np.linalg.norm(x)
    quad = (0.5*0.5-r*r)/(0.5*0.5)
    compute_time = max(0,0.5*quad+x[0])
    time.sleep(compute_time)
    #print('Slept for '+str(compute_time))
    return schwefel([1000*x[0],980*x[1]])[0]


if __name__ == '__main__':
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    SAP = Urgent(slow_and_pointless, bounds=bounds, t_unit=0.025, d_unit=0.3)
    from scipy.optimize import minimize
    res = scipy.optimize.shgo(func=SAP, bounds=bounds, n=8, iters=4, options={'minimize_every_iter':True,'ftol':0.1})

    fig, ax = SAP.mesh_t()
    ax.set_zlabel('$E[t]$')
    fig.show()
    num_requested = res.nfev
    num_computed  = len(SAP.cache_y)
    cpu_time = np.sum(SAP.cache_t)
    try:
        skipped_time = np.sum(SAP.approx_t)
        saving = skipped_time/(skipped_time+cpu_time)
        from pprint import pprint
        pprint(SAP.approx_message)
    except:
        pass
    print('Found '+str(res.fun))
    pass
    fig, ax = SAP.mesh_y()
    ax.set_zlabel('f')
    fig.show()
    pass



