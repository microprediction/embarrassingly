from embarrassingly.parallel import Parallel
import optuna
import time

# Use optuna in parallel fashion without the need for a database

# See https://www.microprediction.com/blog/robust-optimization for explanation

def pre_objective(worker, trial):
    print('Hi this is worker ' + str(worker)+' taking 5 seconds.')
    x = [trial.suggest_float('x' + str(i), 0, 1) for i in range(3)]
    time.sleep(5)
    return x[0] + x[1] * x[2]


def optimize():
    start_time = time.time()
    objective = Parallel(pre_objective, num_workers=7)
    study = optuna.create_study()
    study.optimize(objective, n_trials=21, n_jobs=7)
    end_time = time.time()
    print(end_time-start_time)


if __name__=='__main__':
    optimize()