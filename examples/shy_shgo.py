from embarrassingly.shy import Shy, slow_and_pointless
from scipy.optimize import shgo
import numpy as np
import matplotlib.pyplot as plt

labels = list()
bounds = [(-0.5, 0.5), (-0.5, 0.5)]
for t_unit in np.logspace(-4,-2,5):
    for d_unit in [0.2]:
        SAP = Shy(slow_and_pointless, bounds=bounds, t_unit=t_unit, d_unit=d_unit)
        res = shgo(func=SAP, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.000001})
        plt.plot(SAP.found_t, SAP.found_func)
        plt.xlabel('CPU Time')
        plt.ylabel('Minimum')
        plt.yscale('log')
        label = "t="+str(round(t_unit,4))+' d='+str(round(d_unit,2))
        labels.append(label)
        plt.legend(labels)
        plt.pause(0.00001)
pass

