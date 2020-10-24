# embarrassingly
parallel

### Example

    from embarrassingly.parallel import Parallel
    import optuna

    def pre_objective(worker, trial):
        print('Hi this is worker ' + str(worker))
        x = [trial.suggest_float('x' + str(i), 0, 1) for i in range(3)]
        return x[0] + x[1] * x[2]
    
    def test_optuna():
        objective = Parallel(pre_objective, num_workers=7)
        study = optuna.create_study()
        study.optimize(objective, n_trials=15)
