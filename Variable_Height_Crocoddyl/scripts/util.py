from math import cos, sin

import numpy as np


def rotx(theta):
    Rx = np.array([[1, 0, 0],
                   [0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]])
    return Rx


def roty(theta):
    Ry = np.array([[cos(theta), 0, sin(theta)],
                   [0, 1, 0],
                   [-sin(theta), 0, cos(theta)]])
    return Ry


def rotz(theta):
    Rz = np.array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta), 0],
                   [0, 0, 1]])
    return Rz
