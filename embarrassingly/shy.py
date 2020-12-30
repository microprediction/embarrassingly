from smt.surrogate_models import IDW
from embarrassingly.scatterbrained import mesh2d
import scipy
from deap.benchmarks import h1, schwefel
from embarrassingly.fastidious import Fastidious
from smt.surrogate_models.surrogate_model import SurrogateModel
import matplotlib.pyplot as plt
import numpy as np
import time

# See https://www.microprediction.com/blog/robust-optimization for explanation


class Shy(Fastidious):

    """ Trick global optimizers into not wasting time by sometimes
        returning a surrogate instead of performing a function evaluation.

        Also includes an option to lie when we do evaluate, for example by
        providing a value lying between the prior surrogate and evaluated value
    """

    # Two key functions to override are listed first.

    def accept(self, d_ratio=None, t_ratio=None)->bool:
        """ Shall we perform evaluation?
           :param d_ratio   Normalized distance from nearest evaluated point
           :param t_ratio   Normalized anticipated time of computation
        """
        if t_ratio is None and d_ratio is None:
            return True
        if t_ratio is None:
            acceptance_probability = d_ratio/(1+d_ratio)
        elif d_ratio is None:
            acceptance_probability = 1/(t_ratio+self.eta)
        else:
            acceptance_probability = d_ratio/((t_ratio+self.eta)*(1+d_ratio))
        return np.random.rand() < acceptance_probability

    def response(self, y_hat, y=None)->float:
        """ By default this returns a Kalman-like response which is reluctant to
            tell the optimizer and extreme value, at least at first.

              y_hat : Surrogate function value
              y     : Actual evaluated func value

        """
        return y_hat + self.kappa * (y - y_hat) if y is not None else y_hat

    def __init__(self, func, bounds, func_args=None, func_kwargs=None,
                 surrogate_model:SurrogateModel=None,
                 cpu_model:SurrogateModel=None,
                 t_unit: float = None, d_unit: float = -1,
                 kappa:float=1.0, eta:float=1e-3):
        """
        :param func:
        :param bounds:          List of pairs of upper/lower bounds
        :param t_unit:          Time scale in seconds, used to normalize before calling self.accept()
        :param d_unit:          Distance scale, used to normalize before calling self.accept(), defaults to diagonal
        :param func_args:       Additional args as tuple
        :param func_kwargs      Additional keyword args as dict
        :param surrogate_model  Something from the Surrogate Modeling Toolbox, to track function itself
        :param cpu_model        Something from the Surrogate Modeling Toolbox, to track cpu times
        :param kappa            Similar role to a Kalman gain. Can help plateau search.
        :param eta              Constant determining acceptance probability as function of time. See accept()


        """
        super().__init__(func=func, func_args=func_args, func_kwargs=func_kwargs)
        self.dim = len(bounds)
        self.kappa = kappa
        self.eta = eta
        self.bounds = bounds
        self.verbose = True
        self.faketime = False
        self.diameter = np.linalg.norm([b[1] - b[0] for b in bounds])
        self.surrogate_model = IDW(p=2) if surrogate_model is None else surrogate_model     # Surrogate function
        self.surrogate_model.options.update({'print_global': False})
        self.cpu_model = IDW(p=2) if cpu_model is None else cpu_model             # Function estimating computation time
        self.cpu_model.options.update({'print_global': False})
        self.t_unit = t_unit
        if d_unit is not None and d_unit < 0:
            self.d_unit = 0.1 * np.linalg.norm([b[1] - b[0] for b in self.bounds])
        else:
            self.d_unit = d_unit
        self.xs, self.ys, self.ts, self.t_hats, self.y_hats, self.y_responses, self.log = list(), list(), list(), list(), list(), list(), list()


    def call_cpu_model(self, x:np.array):
        return self.cpu_model.predict_values(np.array([x]))[0][0]

    def call_surrogate_model(self, x: np.array):
        return self.surrogate_model.predict_values(np.array([x]))[0][0]

    def log_response(self, x, t, t_hat, y_hat, y_response, y=np.nan):
        """ Whether or not evaluation is performed """
        # Parent stores training data separately. Some redundancy for convenience.
        self.xs.append(np.array(x))
        self.ys.append(np.array(y))
        self.ts.append(np.array(t))
        self.t_hats.append(np.array(t_hat))
        self.y_hats.append(np.array(y_hat))
        self.y_responses.append(y_response)

    def _timed_call_with_surrogate_training(self, x, *args, **kwargs):
        """
        :param x:          np.array typically
        :param args:       Additional args to self.func Don't supply if you instantiated with args or kwargs already
        :param kwargs:
        :return:    value, cpu_time
        """
        y,t = self._call_fastidious(x=x,*args,**kwargs)
        self.cpu_model.set_training_values(np.array(self.train_x), np.array(self.train_t))
        self.cpu_model.train()
        self.surrogate_model.set_training_values(np.array(self.train_x), np.array(self.train_y))
        self.surrogate_model.train()
        if self.found_y is None or y < self.found_y[-1]:
            total_cpu = np.sum(self.train_t)
            self.log_new_minima(x=x, y=y, t=total_cpu)
            if self.verbose:
                print('New min '+str(y)+' at '+str(x))
        return y, t

    def __call__(self, x, *args, **kwargs):
        """ Sometimes evaluate, but other times return surrogate instead
            Always updates the surrogate model, and cpu model
        """
        x = np.asarray(x)
        if self.train_x is None:
            y, t = self._timed_call_with_surrogate_training(x, *args, **kwargs)
            y_hat = y
            y_response = y
            t_hat = 0
        else:
            t_hat = self.call_cpu_model(x=x)  # Always call, just for good stats
            d = self.distance_to_cache(x=x, default_dist=self.diameter)
            t_ratio = t_hat/self.t_unit if self.t_unit is not None and self.t_unit>0 else None
            d_ratio = d/self.d_unit if self.d_unit is not None and self.d_unit>0 else None
            y_hat = self.call_surrogate_model(x=x)
            if self.accept(d_ratio=d_ratio, t_ratio=t_ratio):
                y, t = self._timed_call_with_surrogate_training(x, *args, **kwargs)
                y_response = self.response(y=y, y_hat=y_hat)
            else:
                y_response = y_hat
                t = np.nan
                y = np.nan
        self.log_response(x=x, t=t, y=y, y_response=y_response, y_hat=y_hat, t_hat=t_hat)
        return y_response


class Shy2d(Shy):

    # Adds visuals for 2d functions

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def cpu_surrogate_surface_plot(self):
        """ Show (present) CPU surrogate model, and training data """
        # Obviously will only work for 2d
        fig, ax = mesh2d(f=self.call_cpu_model, bounds=self.bounds)
        x1 = [x[0] for x in self.train_x]
        x2 = [x[1] for x in self.train_x]
        fs = [y for y in self.train_t]
        ax.scatter(xs=x1, ys=x2, zs=fs, c='blue')
        # Also show in green the contemporaneous estimates of cpu time
        # using historical surrogate values
        if self.xs is not None:
            x1 = [x[0] for x in self.xs]
            x2 = [x[1] for x in self.xs]
            ts = [y for y in self.t_hats]
            ax.scatter(xs=x1, ys=x2, zs=ts, c='green')

        ax.set_title("CPU model")
        fig.show()
        return fig, ax

    def underpromotion_surface_plot(self):
        evaluated = [ not np.isnan(y) for y in self.ys ]

        x1 = [x[0] for x,e in zip(self.xs, evaluated) if e]
        x2 = [x[1] for x,e in zip(self.xs, evaluated) if e]
        y_eval = [y for y, e in zip(self.ys, evaluated) if e]
        y_response = [y for y, e in zip(self.y_responses, evaluated) if e]
        y_under = [ abs(y_e-y_r) for y_e, y_r in zip(y_eval,y_response)]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=x1, ys=x2, zs=y_under, c='blue', alpha=0.9)
        return fig, ax


    def surrogate_surface_plot(self, show_training=None, show_contemporaneous=False, show_responses=False):
        """ Show (present) surrogate function and actual evaluations """
        fig, ax = mesh2d(f=self.call_surrogate_model, bounds=self.bounds)
        x1 = [x[0] for x in self.train_x]
        x2 = [x[1] for x in self.train_x]
        fs = [y for y in self.train_y]
        if show_training is None:
            show_training = not show_responses

        if show_responses:
            try:
                x1 = [x[0] for x in self.xs]
                x2 = [x[1] for x in self.xs]
                fs = [y for y in self.y_responses]
                ax.scatter(xs=x1, ys=x2, zs=fs, c='magenta',alpha=0.9)
            except TypeError:
                pass

        if show_contemporaneous:
            try:
                x1 = [x[0] for x in self.xs]
                x2 = [x[1] for x in self.xs]
                fs = [y for y in self.y_hats]
                ax.scatter(xs=x1, ys=x2, zs=fs, c='green',alpha=0.9)
            except TypeError:
                pass

        if show_training:
            ax.scatter(xs=x1, ys=x2, zs=fs, c='blue', alpha=0.9)

        title = 'Surrogate model ' + ' and responses ' if show_responses else ''
        ax.set_title(title)
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

def slow_and_pointless_and_fast(x):
    r = np.linalg.norm(x)
    quad = (0.5 * 0.5 - r * r) / (0.5 * 0.5)
    compute_time = max(0, 0.5 * quad + x[0])
    return schwefel([1000 * x[0], 980 * x[1]])[0], compute_time

def approx_demo():
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    f = slow_and_pointless
    f_tilde = Shy2d(f, bounds=bounds, t_unit=0.01, d_unit=0.3, kappa=0.5)
    from scipy.optimize import minimize
    res = scipy.optimize.shgo(func=f_tilde, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})

    fig, ax = f_tilde.cpu_surrogate_surface_plot()
    ax.set_zlabel('$E[t]$')
    fig.show()

    fig, ax = f_tilde.surrogate_surface_plot(show_responses=True)
    ax.set_zlabel('f')
    fig.show()
    pass

    fig, ax = f_tilde.underpromotion_surface_plot()
    ax.set_zlabel('f')
    fig.show()
    pass

if __name__ == '__main__':
    approx_demo()
    pass




