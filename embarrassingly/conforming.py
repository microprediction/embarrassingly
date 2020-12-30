
# This little module makes some other optimizers look like scipy.optimize.sgho

# See https://www.microprediction.com/blog/robust-optimization for explanation


import optuna
import numpy as np


def optuna_minimize(func, bounds, args=(), n:int=100 ):
    """
          func   callable
          bounds           [ (-1,1), (-1,1) ]
    """
    dim = len(bounds)
    def optuna_style_objective(trial):
        x = [trial.suggest_uniform('x' + str(i), bounds[i][0], bounds[i][1]) for i in range(dim)]
        return func(x,*args)

    study = optuna.create_study()
    study.optimize(optuna_style_objective, n_trials=n)
    x = np.array([ study.best_params['x'+str(i)] for i in range(dim) ])
    fun = study.best_value
    nfev = len(study.trials)
    return dict(x=x, fun=fun, nfev=nfev)

