import crocoddyl
import numpy as np
from math import sqrt
from util import rotRollPitchYaw


class AsymmetricFrictionConeResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, ori, mu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu, True, False, True)
        self.friction_x_p = np.zeros(3)
        self.friction_x_n = np.zeros(3)
        self.friction_y_p = np.zeros(3)
        self.friction_y_n = np.zeros(3)
        self.ori = ori
        self.mu = mu
        self.compute_friction_cones()

    def compute_friction_cones(self):
        x_p = np.array([self.mu / (sqrt(self.mu ** 2 + 1)), 0, 1 / (sqrt(self.mu ** 2 + 1))])
        x_n = np.array([-self.mu / (sqrt(self.mu ** 2 + 1)), 0, 1 / (sqrt(self.mu ** 2 + 1))])
        y_p = np.array([0, self.mu / (sqrt(self.mu ** 2 + 1)), 1 / (sqrt(self.mu ** 2 + 1))])
        y_n = np.array([0, -self.mu / (sqrt(self.mu ** 2 + 1)), 1 / (sqrt(self.mu ** 2 + 1))])
        R_L = rotRollPitchYaw(self.ori[0], self.ori[1], self.ori[2])
        R_R = rotRollPitchYaw(self.ori[3], self.ori[4], self.ori[5])
        x_p_l = R_L.dot(x_p)
        x_n_l = R_L.dot(x_n)
        y_p_l = R_L.dot(y_p)
        y_n_l = R_L.dot(y_n)

        x_p_r = R_R.dot(x_p)
        x_n_r = R_R.dot(x_n)
        y_p_r = R_R.dot(y_p)
        y_n_r = R_R.dot(y_n)

        if x_p_r[2] > x_p_l[2]:
            self.friction_x_p = x_p_l
        else:
            self.friction_x_p = x_p_r
        if x_n_r[2] > x_n_l[2]:
            self.friction_x_n = x_n_l
        else:
            self.friction_x_n = x_n_r
        if y_p_r[2] > y_p_l[2]:
            self.friction_y_p = y_p_l
        else:
            self.friction_y_p = y_p_r
        if y_n_r[2] > y_n_l[2]:
            self.friction_y_n = y_n_l
        else:
            self.friction_y_n = y_n_r

    def calc(self, data, x, u):
        data.p[:] = x[:3] - data.shared.actuation.tau
        data.f[0] = data.p[0] / data.p[2]
        data.f[1] = data.p[1] / data.p[2]
        data.r[0] = -self.friction_x_p.dot(data.f)
        data.r[1] = -self.friction_x_n.dot(data.f)
        data.r[2] = -self.friction_y_p.dot(data.f)
        data.r[3] = -self.friction_y_n.dot(data.f)

    def calcDiff(self, data, x, u):
        data.Rx[:, :3] = \
            -np.array([[self.friction_x_p[0] / data.p[2], self.friction_x_p[1] / data.p[2], -self.friction_x_p[0] * data.p[0] / (data.p[2]**2) - self.friction_x_p[1] * data.p[1] / (data.p[2])**2],
                       [self.friction_x_n[0] / data.p[2], self.friction_x_n[1] / data.p[2], -self.friction_x_n[0] * data.p[0] / (data.p[2]**2) - self.friction_x_n[1] * data.p[1] / (data.p[2])**2],
                       [self.friction_y_p[0] / data.p[2], self.friction_y_p[1] / data.p[2], -self.friction_y_p[0] * data.p[0] / (data.p[2]**2) - self.friction_y_p[1] * data.p[1] / (data.p[2])**2],
                       [self.friction_y_n[0] / data.p[2], self.friction_y_n[1] / data.p[2], -self.friction_y_n[0] * data.p[0] / (data.p[2]**2) - self.friction_y_n[1] * data.p[1] / (data.p[2])**2]])
        data.dr_dtau[:, :] = \
            -np.array([[-self.friction_x_p[0] / data.p[2], -self.friction_x_p[1] / data.p[2], self.friction_x_p[0] * data.p[0] / (data.p[2])**2 + self.friction_x_p[1] * data.p[1] / (data.p[2])**2],
                       [-self.friction_x_n[0] / data.p[2], -self.friction_x_n[1] / data.p[2], self.friction_x_n[0] * data.p[0] / (data.p[2])**2 + self.friction_x_n[1] * data.p[1] / (data.p[2])**2],
                       [-self.friction_y_p[0] / data.p[2], -self.friction_y_p[1] / data.p[2], self.friction_y_p[0] * data.p[0] / (data.p[2])**2 + self.friction_y_p[1] * data.p[1] / (data.p[2])**2],
                       [-self.friction_y_n[0] / data.p[2], -self.friction_y_n[1] / data.p[2], self.friction_y_n[0] * data.p[0] / (data.p[2])**2 + self.friction_y_n[1] * data.p[1] / (data.p[2])**2]])
        data.Ru[:,:] = data.dr_dtau.dot(data.shared.actuation.dtau_du)

    def createData(self, collector):
        return AsymmetricFrictionConeDataResidual(self, collector)

class AsymmetricFrictionConeDataResidual(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        self.dr_dtau = np.zeros((4,3))
        self.p = np.zeros(3)
        self.f = np.zeros(3)
        self.f[2] = 1.

