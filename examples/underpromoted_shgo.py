from scipy.optimize import shgo
from embarrassingly.underpromoted import plateaudinous, Underpromoted2d


# Where do we land?


if __name__=='__main__':
    bounds = [(-1 ,1) ,(-1 ,1)]
    f = plateaudinous
    res1 = shgo(func=f, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Minimum at '+str(res1.x))

    f_tilde = Underpromoted2d(f, bounds=bounds, radius=0.05)
    res1 = shgo(func=f_tilde, bounds=bounds, n=8, iters=4, options={'minimize_every_iter': True, 'ftol': 0.1})
    print('Landed at '+str(res1.x))