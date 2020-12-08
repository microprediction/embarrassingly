import numpy as np
import math
from embarrassingly.plot_util import mesh2d
import matplotlib.pyplot as plt

# A very sophisticated (ahem) model for strokes around the green


def putt(xs)->float:
    """
    :param xs:
    :return:
    """
    distance = np.linalg.norm(xs)
    make_prob = math.exp(-distance/12)
    three_prob = max( math.exp(0.01*(distance-30)), 0.2 )
    two_prob = 1-make_prob-three_prob
    return make_prob + 2*two_prob + 3*three_prob


def bunker(xs)->float:
    """
    :param xs:
    :return:
    """
    distance = np.linalg.norm(xs)
    make_prob = 0.02*math.exp(-distance/5)
    three_prob = max( math.exp(0.015 * (distance)) , 0.7)
    two_prob = 1 - make_prob - three_prob
    return make_prob + 2*two_prob + 3*three_prob


def rough(xs) -> float:
    """
    :param xs:
    :return:
    """
    distance = min(np.linalg.norm(xs),10)
    make_prob = math.exp(-distance/9)
    three_prob = max(math.exp(0.015 * (distance - 10)), 0.4)
    two_prob = 1 - make_prob - three_prob
    return make_prob + 2 * two_prob + 3 * three_prob


def shots(xs)->float:
    """
    :param xs:    Two-dimensional position on putting green
    :return:      Expected number of shots to finish hole
    """
    bunkers = [ (np.array([-0.5,-0.55]), 0.25),
                (np.array([0,1.85]), 0.8) ]
    greens = [  (np.array([0, 0]), 0.7),
                (np.array([-1.5,0]), 1.5)]

    in_bunker = any( np.linalg.norm( xs-b[0] )<b[1] for b in bunkers )
    on_green = any( np.linalg.norm( xs-g[0] )<g[1] for g in greens )
    if in_bunker:
        return bunker(xs)
    elif on_green:
        return putt(xs)
    else:
        return rough(xs)


SPRAY = [ np.array([0,0]),
          np.array([0,1]),
          np.array([1,0]),
          np.array([-1,0]),
          np.array([0,-1])]

def robot(xs)->float:
    """
    A farce
    :param xs:  Robot parameters
    :return:
    """
    arm_rotation  = xs[0]
    body_rotation = xs[1]
    power = arm_rotation+body_rotation
    wobble = (arm_rotation-5)**2 - (body_rotation-5)**2
    roll = math.exp(-(10+power)/10)
    ball = np.array([5*math.exp(wobble)-1,roll])   # Roll past the hole, slice

    scores = list()
    for multiplier in [0.02,0.04,0.06]:
        for spry in SPRAY:
            z = ball + spry*multiplier
            scr = shots(z)
            scores.append(scr)
    return float(np.mean(scores))


def plot_green():
    mesh2d(shots, bounds=[(-3,3),(-3,3)],resolution=0.02 )


def plot_robot():
    mesh2d(robot, bounds=[(-2,5),(-2,5)], resolution=0.027)
    plt.xlabel('Arms')
    plt.ylabel('Body')
    plt.title('Golf robot objective function')
    plt.show()


if __name__=="__main__":
    robot(np.array([3.1,3]))
    plot_robot()
    pass

