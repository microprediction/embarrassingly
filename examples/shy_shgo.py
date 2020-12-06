from embarrassingly.shy import Shy, slow_and_pointless
from scipy.optimize import shgo
import matplotlib.pyplot as plt

labels = list()
bounds = [(-0.5, 0.5), (-0.5, 0.5)]

# See https://www.microprediction.com/blog/robust-optimization for explanation

def rround(x,ndigits):
    try:
        return str(round(x,ndigits=ndigits))
    except:
        return ''

for t_unit in [None, 0.2, 0.3, 0.4]:
    for d_unit in [None, 0.01, 0.02, 0.03]:
        SAP = Shy(slow_and_pointless, bounds=bounds, t_unit=t_unit, d_unit=d_unit)
        res = shgo(func=SAP, bounds=bounds, n=10, iters=4, options={'minimize_every_iter': False, 'ftol': 0.00001})
        plt.plot(SAP.found_tau, SAP.found_y)
        plt.xlabel('CPU Time')
        plt.ylabel('Minimum')
        plt.yscale('log')
        plt.pause(0.00001)
        label = "$\delta t="+rround(t_unit,4)+'$ $\delta d='+rround(d_unit,3)+'$'
        labels.append(label)
        plt.legend(labels)
        plt.pause(0.00001)
        pass
pass

