from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt


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


def plotComMotion(xs, us):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cx = [x[0] for x in xs]
    cy = [x[1] for x in xs]
    cz = [x[2] for x in xs]
    cxdot = [x[3] for x in xs]
    cydot = [x[4] for x in xs]
    czdot = [x[5] for x in xs]
    f_z = [u[0] for u in us]
    # u_x = [u[1] for u in us]
    # u_y = [u[2] for u in us]
    # u_z = [u[3] for u in us]

    plt.plot(cx)
    plt.show()
    plt.plot(cy)
    plt.show()
    plt.plot(cz)
    plt.show()
    # plt.plot(u_x)
    # plt.show()
    # plt.plot(u_y)
    # plt.show()
