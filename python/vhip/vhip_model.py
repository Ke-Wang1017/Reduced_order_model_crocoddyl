import crocoddyl
import numpy as np
import pinocchio


class DifferentialActionModelVariableHeightPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costs, actuation):  # pass vertex class
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, 8)  # cannot overwrite the function, nu = 8: f_z, vertex points(7)

        self._m = pinocchio.computeTotalMass(self.state.pinocchio)
        self._g = abs(self.state.pinocchio.gravity.linear[2])
        self.costs = costs
        self.actuation = actuation  # need to update vertex and create data
        self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([2.0 * self._m * self._g, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def get_support_index(self, support_index):
        if support_index == 1:  # left support
            self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.u_ub = np.array([2.0 * self._m * self._g, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        if support_index == -1:  # right support
            self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.u_ub = np.array([2.0 * self._m * self._g, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def calc(self, data, x, u):
        # Compute actuation
        self.actuation.calc(data.actuation, x, u)
        # Compute dynamics
        data.p[:] = x[:3] - data.actuation.tau
        f_z = u[0]
        data.hm = data.p[2] * self._m
        data.xout[0] = f_z * data.p[0] / data.hm
        data.xout[1] = f_z * data.p[1] / data.hm
        data.xout[2] = f_z / self._m - self._g
        # Computes cost
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        # Derivatives of actuation
        self.actuation.calcDiff(data.actuation, x, u)
        # Derivatives of dynamics
        f_z = u[0]

        data.Fx[0, 0] = f_z / data.hm
        data.Fx[0, 2] = -f_z * data.p[0] / (data.hm * data.p[2])
        data.Fx[1, 1:3] = [f_z / data.hm, -f_z * data.p[1] / (data.hm * data.p[2])]

        data.tmp_hm2 = data.hm * data.hm
        data.az = self._m * f_z
        data.df_dtau[0, 0] = -f_z / data.hm
        data.df_dtau[0, 2] = data.az * data.p[0] / data.tmp_hm2
        data.df_dtau[1, 1:3] = [-f_z / data.hm, data.az * data.p[1] / data.tmp_hm2]

        data.Fu[:, :] = data.df_dtau.dot(data.actuation.dtau_du)
        data.Fu[0, 0] += data.p[0] / data.hm
        data.Fu[1, 0] += data.p[1] / data.hm
        data.Fu[2, 0] += 1. / self._m
        # Derivatives of cost functions
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataVariableHeightPendulum(self)


class DifferentialActionDataVariableHeightPendulum(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.actuation = model.actuation.createData()
        self.collector = crocoddyl.DataCollectorActuation(self.actuation)
        self.costs = model.costs.createData(self.collector)
        self.costs.shareMemory(self)
        self.df_dtau = np.zeros((3, 3))
        self.p = np.zeros(3)  # difference better CoM and CoP
        self.hm = 0.  # pendulum height times mass
        self.az = 0.  # vertical force times mass
        self.tmp_hm2 = 0.