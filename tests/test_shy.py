from embarrassingly.shy import Shy
import numpy as np


def test_response():
    f = lambda xs: np.linalg.norm(xs)
    bounds = [(-1,1),(-1,1)]
    f_tilde = Shy(f,bounds=bounds, kappa=0.5)
    y_response = f_tilde.response(y_hat=1.0, y=2.0)
    assert y_response > 1.0
    assert y_response < 2.0
