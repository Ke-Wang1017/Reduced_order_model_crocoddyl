import crocoddyl
import numpy as np
from math import sqrt
from util import rotRollPitchYaw


class AsymmetricFrictionConeResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, cone):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu, True, False, True)
        self.cone = cone

    def calc(self, data, x, u):
        data.p[:] = x[:3] - data.shared.actuation.tau
        data.f[0] = data.p[0] / data.p[2]
        data.f[1] = data.p[1] / data.p[2]
        data.r[0] = -self.cone.xp.dot(data.f)
        data.r[1] = -self.cone.xn.dot(data.f)
        data.r[2] = -self.cone.yp.dot(data.f)
        data.r[3] = -self.cone.yn.dot(data.f)

    def calcDiff(self, data, x, u):
        xn, xp, yn, yp = self.cone.xn, self.cone.xp, self.cone.yn, self.cone.yp
        data.tmp_pz2 = data.p[2]**2
        data.Rx[:, :3] = \
            -np.array([[xp[0] / data.p[2], xp[1] / data.p[2], -xp[0] * data.p[0] / data.tmp_pz2 - xp[1] * data.p[1] / data.tmp_pz2],
                       [xn[0] / data.p[2], xn[1] / data.p[2], -xn[0] * data.p[0] / data.tmp_pz2 - xn[1] * data.p[1] / data.tmp_pz2],
                       [yp[0] / data.p[2], yp[1] / data.p[2], -yp[0] * data.p[0] / data.tmp_pz2 - yp[1] * data.p[1] / data.tmp_pz2],
                       [yn[0] / data.p[2], yn[1] / data.p[2], -yn[0] * data.p[0] / data.tmp_pz2 - yn[1] * data.p[1] / data.tmp_pz2]])
        data.Ru[:, :] = -data.Rx[:, :3].dot(data.shared.actuation.dtau_du)

    def createData(self, collector):
        return AsymmetricFrictionConeDataResidual(self, collector)


class AsymmetricFrictionConeDataResidual(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        self.p = np.zeros(3)
        self.f = np.zeros(3)
        self.f[2] = 1.
        self.tmp_pz2 = 0.


class AsymmetricCone:
    def __init__(self, mu, ori):
        self.mu = mu
        self.ori = ori
        self._updateCone()
        self.xp = np.zeros(3)
        self.xn = np.zeros(3)
        self.yp = np.zeros(3)
        self.xn = np.zeros(3)

    def set_cone(self, mu, ori):
        self.mu = mu
        self.ori = ori
        self._updateCone()

    def _updateCone(self):
        xp = np.array([self.mu / (sqrt(self.mu**2 + 1)), 0, 1 / (sqrt(self.mu**2 + 1))])
        xn = np.array([-self.mu / (sqrt(self.mu**2 + 1)), 0, 1 / (sqrt(self.mu**2 + 1))])
        yp = np.array([0, self.mu / (sqrt(self.mu**2 + 1)), 1 / (sqrt(self.mu**2 + 1))])
        yn = np.array([0, -self.mu / (sqrt(self.mu**2 + 1)), 1 / (sqrt(self.mu**2 + 1))])

        R_l = rotRollPitchYaw(self.ori[0], self.ori[1], self.ori[2])
        R_r = rotRollPitchYaw(self.ori[3], self.ori[4], self.ori[5])
        xp_l = R_l.dot(xp)
        xn_l = R_l.dot(xn)
        yp_l = R_l.dot(yp)
        yn_l = R_l.dot(yn)

        xp_r = R_r.dot(xp)
        xn_r = R_r.dot(xn)
        yp_r = R_r.dot(yp)
        yn_r = R_r.dot(yn)

        if xp_r[2] > xp_l[2]:
            self.xp = xp_l
        else:
            self.xp = xp_r
        if xn_r[2] > xn_l[2]:
            self.xn = xn_l
        else:
            self.xn = xn_r
        if yp_r[2] > yp_l[2]:
            self.yp = yp_l
        else:
            self.yp = yp_r
        if yn_r[2] > yn_l[2]:
            self.yn = yn_l
        else:
            self.yn = yn_r
