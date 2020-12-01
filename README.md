# embarrassingly

Embarrassingly obvious (in retrospect) ways to hack objective functions before you send them to optimization routines. 

### Example 1 : Parallel objective computation 

    from embarrassingly.parallel import Parallel
    import optuna

    def pre_objective(worker, trial):
        print('Hi this is worker ' + str(worker))
        x = [trial.suggest_float('x' + str(i), 0, 1) for i in range(3)]
        return x[0] + x[1] * x[2]
    
    def test_optuna():
        objective = Parallel(pre_objective, num_workers=7)
        study = optuna.create_study()
        study.optimize(objective, n_trials=15, n_jobs=7)

### Example 2 : Plateau finding

    from embarrassingly.cautious import Cautious
    import numpy as np
    import math
    from scipy.optimize import shgo

    def plateaudinous(x):
    """ A helicopter landing pad when you turn it upside down """
    r = np.linalg.norm(x)
    x0 = np.array([0.25,0.25])
    amp = r*math.sin(16*r*r)
    return -1 if np.linalg.norm(x-x0)<0.1 else 0.1*x[0] + amp
    
    bounds = [(-1,1),(-1,1)]
    res1 = shgo(func=plateaudinous, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print("Global min occurs at "+str(res1.x))

    # But let's land our helicopter in the flat spot! 
    platypus = Cautious(plateaudinous, bounds=bounds, radius=0.01)
    res2 = shgo(func=platypus, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Helicopter lands at '+str(res2.x))


### Example 3 : Expensive functions 

    See https://github.com/microprediction/embarrassingly/blob/main/examples/shy_shgo.py
    
