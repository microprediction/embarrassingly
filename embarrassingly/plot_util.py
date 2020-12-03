
import numpy as np
import matplotlib.pyplot as plt

RESOLUTION = 0.01    # Used 0.003 for plots


def mesh2d(f,bounds):
    """ Plot function taking len 2 vector as single argument
          f(xs)
        Also put points on
    """
    def g(x,y):
        return f(np.array([x,y]))
    return mesh2d_xy(g,bounds)


def mesh2d_xy(f,bounds):
    """ Plot function taking two arguments
        f(x,y)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = np.arange( bounds[0][0], bounds[0][1], RESOLUTION )
    x2 = np.arange(bounds[0][0], bounds[0][1], RESOLUTION)

    X1, X2 = np.meshgrid(x1, x2)
    zs = np.array([ f(x1_,x2_) for x1_,x2_ in zip( np.ravel(X1), np.ravel(X2)) ])
    Z = zs.reshape(X1.shape)

    ax.plot_surface(X1, X2, Z, alpha=0.7)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    return fig, ax