from math import cos, sin
import numpy as np


def rotx(theta):
    Rx = np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
    return Rx


def roty(theta):
    Ry = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    return Ry


def rotz(theta):
    Rz = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    return Rz


def rotRollPitchYaw(roll, pitch, yaw):  # R_yaw*R_pitch*R_roll
    R = np.array([[
        cos(yaw) * cos(pitch),
        cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll),
        cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)
    ],
                  [
                      sin(yaw) * cos(pitch),
                      sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll),
                      sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)
                  ], [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll)]])
    return R
