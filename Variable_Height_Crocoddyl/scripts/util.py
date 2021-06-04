import numpy as np
from math import cos,sin

def rotx(theta):
    Rx = np.vstack(np.hstack(1, 0, 0),
                 np.hstack(0, cos(theta), -sin(theta)),
                 np.hstack(0, sin(theta), cos(theta)))
    return Rx


def roty(theta):
    Ry = np.vstack(np.hstack(cos(theta), 0, sin(theta)),
                 np.hstack(0, 1, 0),
                 np.hstack(-sin(theta), 0, cos(theta)))
    return Ry


def rotz(theta):
    Rz = np.vstack(np.hstack(cos(theta), -sin(theta), 0),
                 np.hstack(sin(theta), cos(theta), 0),
                 np.hstack(0, 0, 1))
    return Rz