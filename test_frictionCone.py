from util import rotRollPitchYaw, rotx
import numpy as np
from math import sqrt

mu = 0.7
x_p = np.array([mu / (sqrt(mu ** 2 + 1)), 0, 1 / (sqrt(mu ** 2 + 1))])
x_n = np.array([-mu / (sqrt(mu ** 2 + 1)), 0, 1 / (sqrt(mu ** 2 + 1))])
y_p = np.array([0, mu / (sqrt(mu ** 2 + 1)), 1 / (sqrt(mu ** 2 + 1))])
y_n = np.array([0, -mu / (sqrt(mu ** 2 + 1)), 1 / (sqrt(mu ** 2 + 1))])

ori_L = np.array([15.0, 10., 0.])
ori_R = np.array([-15.0, 10., 0.])
R_L = rotRollPitchYaw(ori_L[0], ori_L[1], ori_L[2])
R_R = rotRollPitchYaw(ori_R[0], ori_R[1], ori_R[2])
# R_L = rotx(ori_L[0])
# R_R = rotx(ori_R[0])

x_p_l = R_L.dot(x_p)
x_n_l = R_L.dot(x_n)
y_p_l = R_L.dot(y_p)
y_n_l = R_L.dot(y_n)

x_p_r = R_R.dot(x_p)
x_n_r = R_R.dot(x_n)
y_p_r = R_R.dot(y_p)
y_n_r = R_R.dot(y_n)

print('xpr is ', x_p_r)