from smt.surrogate_models import IDW
from embarrassingly.visual import mesh2d
import scipy
from deap.benchmarks import h1, schwefel
from embarrassingly.fastidious import Fastidious
from smt.surrogate_models.surrogate_model import SurrogateModel
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import math


class Shy(Fastidious):

    """ Trick global optimizers into not wasting time by sometimes
        returning a surrogate instead of performing a function evaluation
    """

    # You may wish to override the default acceptance function

    def accept(self, d_ratio, t_ratio)->bool:
        """ Shall we perform evaluation?
           :param d_ratio   Normalized distance from nearest evaluated point
           :param t_ratio   Normalized anticipated time of computation
        """
        acceptance_probability = math.exp(d_ratio)/(0.1+t_ratio)
        return np.random.rand() < acceptance_probability

    def __init__(self, func, bounds, t_unit:float=1., d_unit:float=None, func_args=None, func_kwargs=None,
                 surrogate_model:SurrogateModel=None,
                 cpu_model:SurrogateModel=None):
        """
        :param func:
        :param bounds:          List of pairs of upper/lower bounds
        :param t_unit:          Time scale in seconds, used to normalize before calling self.accept()
        :param d_unit:          Distance scale, used to normalize before calling self.accept(), defaults to diagonal
        :param func_args:       Additional args as tuple
        :param func_kwargs      Additional keyword args as dict
        :param surrogate_model  Something from the Surrogate Modeling Toolbox, to track function itself
        :param cpu_model        Something from the Surrogate Modeling Toolbox, to track cpu times

        """
        super().__init__(func=func, func_args=func_args, func_kwargs=func_kwargs)
        self.dim = len(bounds)
        self.func = func
        self.bounds = bounds
        self.diameter = np.linalg.norm([b[1] - b[0] for b in bounds])
        self.surrogate_model = IDW(p=2) if surrogate_model is None else surrogate_model     # Surrogate function
        self.surrogate_model.options.update({'print_global': False})
        self.cpu_model = IDW(p=2) if cpu_model is None else cpu_model             # Function estimating computation time
        self.cpu_model.options.update({'print_global': False})
        self.t_unit = t_unit
        self.d_unit = d_unit if d_unit is not None else 0.1*np.linalg.norm([b[1] - b[0] for b in bounds])
        self.surrogate_x = None              # Cache for all surrogate estimates returned to user, x-values
        self.surrogate_func = None              # Cache for all surrogate estimates returned to user, surrogate-values
        self.surrogate_messages = None

    def call_cpu_model(self, x:np.array):
        return self.cpu_model.predict_values(np.array([x]))[0][0]

    def call_surrogate_model(self, x: np.array):
        return self.surrogate_model.predict_values(np.array([x]))[0][0]

    def update_approx(self, x, y, t, message = ''):
        """ Update record of approximations returned to user """
        if self.surrogate_x is None:
            self.surrogate_x = [np.array(x)]
            self.surrogate_func = [np.array(y)]
            self.approx_t = [np.array(t)]
            self.surrogate_messages = [message]
        else:
            self.surrogate_x.append(np.array(x))
            self.surrogate_func.append(np.array(y))
            self.approx_t.append(np.array(t))
            self.surrogate_messages.append(message)

    @staticmethod
    def timed_call(f,x,*args,**kwargs):
        start_time = time.time()
        y = f(x,*args,**kwargs)
        t = time.time()-start_time
        return y,t

    def call_and_train(self, x, *args, **kwargs):
        y, t = self.timed_call(self.func,x=x,*args,**kwargs)
        self.update_cache(x=x, y=y, t=t)
        self.cpu_model.set_training_values(np.array(self.cache_x), np.array(self.cache_t))
        self.cpu_model.train()
        self.surrogate_model.set_training_values(np.array(self.cache_x), np.array(self.cache_func))
        self.surrogate_model.train()
        if self.found_func is None or y < self.found_func[-1]:
            total_cpu = np.sum(self.cache_t)
            self.update_found(x=x,y=y,t=total_cpu)
            print('New min '+str(y)+' at '+str(x))
        return y

    def __call__(self, x, *args, **kwargs):
        x = np.asarray(x)
        if self.cache_x is not None:
            d = self.distance_to_cache(x=x, default_dist=self.diameter)
            t_hat = self.call_cpu_model(x=x)
            t_ratio = t_hat/self.t_unit
            d_ratio = d/self.d_unit
            if self.accept(d_ratio=d_ratio, t_ratio=t_ratio):
                return self.call_and_train(x, *args, **kwargs)
            else:
                y_hat = self.call_surrogate_model(x=x)
                self.update_approx(x=x, y=y_hat, t=t_hat,
                                   message=json.dumps({'yHat': y_hat, 't_ratio':t_ratio,'distance':d }))
                return y_hat

        else:
            return self.call_and_train(x,*args,**kwargs)

    def cpu_surface_plot(self):
        """ Show approximate and actual CPU times """
        fig, ax = mesh2d(f=self.call_cpu_model, bounds=self.bounds)
        x1 = [x[0] for x in self.cache_x]
        x2 = [x[1] for x in self.cache_x]
        fs = [y for y in self.cache_t]
        ax.scatter(xs=x1, ys=x2, zs=fs, c='blue')
        if self.surrogate_x is not None:
            x1 = [x[0] for x in self.surrogate_x]
            x2 = [x[1] for x in self.surrogate_x]
            fs = [y for y in self.approx_t]
            ax.scatter(xs=x1, ys=x2, zs=fs, c='green')

        ax.set_title("Estimated (green) and actual (blue) cpu time")
        fig.show()
        return fig, ax

    def func_surface_plot(self):
        """ Show approximate and actual function """
        fig, ax = mesh2d(f=self.call_surrogate_model, bounds=self.bounds)
        x1 = [x[0] for x in self.cache_x]
        x2 = [x[1] for x in self.cache_x]
        fs = [y for y in self.cache_func]
        ax.scatter(xs=x1, ys=x2, zs=fs)
        try:
            x1 = [x[0] for x in self.surrogate_x]
            x2 = [x[1] for x in self.surrogate_x]
            fs = [y for y in self.surrogate_func]
            ax.scatter(xs=x1, ys=x2, zs=fs, c='green')
        except TypeError:
            pass
        ax.set_title('Objective function')
        ax.legend()
        fig.show()
        return fig, ax


def slow_and_pointless(x):
    """ Example of a function with varying computation time """
    r = np.linalg.norm(x)
    quad = (0.5*0.5-r*r)/(0.5*0.5)
    compute_time = max(0,0.5*quad+x[0])
    time.sleep(compute_time)
    return schwefel([1000*x[0],980*x[1]])[0]


def approx_demo():
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    SAP = Shy(slow_and_pointless, bounds=bounds, t_unit=0.01, d_unit=0.3)
    from scipy.optimize import minimize
    res = scipy.optimize.shgo(func=SAP, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})

    fig, ax = SAP.cpu_surface_plot()
    ax.set_zlabel('$E[t]$')
    fig.show()
    num_requested = res.nfev
    num_computed = len(SAP.cache_func)
    cpu_time = np.sum(SAP.cache_t)
    try:
        skipped_time = np.sum(SAP.approx_t)
        saving = skipped_time / (skipped_time + cpu_time)
        from pprint import pprint
        pprint(SAP.surrogate_messages)
    except:
        pass
    print('Found ' + str(res.fun))
    pass
    fig, ax = SAP.func_surface_plot()
    ax.set_zlabel('f')
    fig.show()
    pass


if __name__ == '__main__':
    approx_demo()




