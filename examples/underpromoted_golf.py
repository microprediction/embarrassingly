from embarrassingly.demonstrative.golf import shots, robot
from embarrassingly.underpromoted import Underpromoted2d
from scipy.optimize import shgo

# See https://www.microprediction.com/blog/robust-optimization for explanation of experiments
# Where to aim on the green?

BOUNDS = [(-4, 4), (-4, 4)]
n = 50
iters = 6
options = {'minimize_every_iter': True, 'ftol': 0.0000001}

if __name__ == '__main__':
    res = shgo(func=robot, bounds=BOUNDS, n=n, iters=iters, options=options)
    arms_and_body = res.x
    cautious = Underpromoted2d(func=robot, bounds=BOUNDS, radius=0.1, kappa=0.125)
    res1 = shgo(func=cautious, bounds=BOUNDS, n=n, iters=iters, options=options)
    cautious_arms_and_body = res1.x
    pass
