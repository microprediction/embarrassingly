# embarrassingly

Embarrassingly obvious (in retrospect) ways to hack objective functions before you send them to optimization routines. 
See [blog article](https://www.microprediction.com/blog/robust-optimization) for motivation and explanation

![](https://i.imgur.com/pvcS5AX.png)

### Install 

    pip install embarrassingly 

### Example 1 : Parallel objective computation 

See [optuna_parallel.py](https://github.com/microprediction/embarrassingly/blob/main/examples/optuna_parallel.py)

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

See [underpromoted_shgo.py](https://github.com/microprediction/embarrassingly/blob/main/examples/underpromoted_shgo.py)

    from scipy.optimize import shgo
    from embarrassingly.underpromoted import plateaudinous, Underpromoted2d
    
    bounds = [(-1 ,1) ,(-1 ,1)]
    f = plateaudinous
    res1 = shgo(func=f, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Minimum at '+str(res1.x))

    f_tilde = Underpromoted2d(f, bounds=bounds, radius=0.05)
    res1 = shgo(func=f_tilde, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Landed at '+str(res1.x))

    

### Example 3 : Expensive functions 

See [shy_shgo.py](https://github.com/microprediction/embarrassingly/blob/main/examples/shy_shgo.py)

    def slow_and_pointless(x):
    """ Example of a function with varying computation time """
        r = np.linalg.norm(x)
        quad = (0.5*0.5-r*r)/(0.5*0.5)
        compute_time = max(0,0.5*quad+x[0])
        time.sleep(compute_time)
        return schwefel([1000*x[0],980*x[1]])[0]
    
    # Save time by making it a "shy" objective function
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    SAP = Shy(slow_and_pointless, bounds=bounds, t_unit=0.01, d_unit=0.3)
    from scipy.optimize import minimize
    res = scipy.optimize.shgo(func=SAP, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    
