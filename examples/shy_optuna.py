from embarrassingly.shy import Shy, slow_and_pointless
from scipy.optimize import shgo
import matplotlib.pyplot as plt
import optuna
labels = list()
bounds = [(-0.5, 0.5), (-0.5, 0.5)]
from embarrassingly.shy import Shy
from embarrassingly.conforming import optuna_minimize

# See https://www.microprediction.com/blog/robust-optimization for explanation

def rround(x,ndigits):
    try:
        return str(round(x,ndigits=ndigits))
    except:
        return 'none'


for t_unit in [None, 0.1, 0.2]:
    for d_unit in [None, 0.01, 0.02]:
        f_tilde = Shy(func=slow_and_pointless,t_unit=t_unit,d_unit=d_unit, bounds=bounds)
        res = optuna_minimize(f_tilde, bounds, n=500)
        # Plot progress
        plt.plot(f_tilde.found_tau, f_tilde.found_y)
        plt.xlabel('CPU Time')
        plt.ylabel('Minimum')
        plt.yscale('log')
        plt.pause(0.00001)
        label = "$\delta_t="+rround(t_unit,4)+'$ $\delta_x='+rround(d_unit,3)+'$'
        labels.append(label)
        plt.legend(labels)
        plt.pause(0.00001)
        pass
    pass
pass

