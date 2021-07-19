from math import sin,cos,sqrt
import numpy as np
import matplotlib.pyplot as plt

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

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qx, qy, qz, qw]

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    qx = Q[0]
    qy = Q[1]
    qz = Q[2]
    qw = Q[3]
    # normalization
    n = 1./ sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx *= n
    qy *= n
    qz *= n
    qw *= n

    return np.array(
        [[1. - 2. * qy * qy - 2. * qz * qz, 2. * qx * qy - 2. * qz * qw, 2. * qx * qz + 2. * qy * qw],
         [2. * qx * qy + 2. * qz * qw, 1. - 2. * qx * qx - 2. * qz * qz, 2. * qy * qz - 2. * qx * qw],
         [2. * qx * qz - 2. * qy * qw, 2. * qy * qz + 2. * qx * qw, 1. - 2. * qx * qx - 2. * qy * qy]])


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
