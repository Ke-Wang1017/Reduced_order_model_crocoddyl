import crocoddyl
import numpy as np
import time

class DifferentialActionModelVariableHeightPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), 1) # nu = 1
        self.uNone = np.zeros(self.nu)

        self.m = 95.0
        self.g = 9.81
        self.ref_vel = np.array([0.0, 0.0])
        self.foot_location = np.array([0.0, 0.085, 0.0001])
        self.costs = costs

    def setRefVel(self, refVel):
        self.ref_vel = refVel

    def setFootLocation(self, footLocation):
        self.foot_location = footLocation

    def calc(self, data, x, u):
        if u is None:
            u = self.uNone
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x # how do you know cdot is the diff of c?
        u_x, u_y, u_z = self.foot_location
        f_z = u.item(0)

        cdotdot_x = f_z * (c_x-u_x) / ((c_z - u_z) * self.m)
        cdotdot_y = f_z * (c_y-u_y) / ((c_z - u_z) * self.m)
        cdotdot_z = f_z/self.m - self.g
        data.xout = np.array([cdotdot_x, cdotdot_y, cdotdot_z]).T

        # compute the cost residual
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost
        # data.r = np.matrix(self.costWeights * np.array([f_z, cdot_x-ref_vel_x, cdot_y-ref_vel_y])).T
        # data.cost = .5* np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        u_x, u_y, u_z = self.foot_location
        f_z = u[0].item()

        data.Fx[:, :] = np.array([[f_z/((c_z-u_z)*self.m), 0.0, -f_z*(c_x-u_x)/((c_z-u_z)**2*self.m), 0.0, 0.0, 0.0],
                                  [0.0, f_z/((c_z-u_z)*self.m), -f_z*(c_y-u_y)/((c_z-u_z)**2*self.m), 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        data.Fu[:] = np.array([(c_x-u_x)/(self.m*(c_z-u_z)), (c_y-u_y)/(self.m*(c_z-u_z)), 1.0/self.m])
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataVariableHeightPendulum(self)


class DifferentialActionDataVariableHeightPendulum(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        shared_data = crocoddyl.DataCollectorAbstract()
        self.costs = model.costs.createData(shared_data)
        self.costs.shareMemory(self)


state = crocoddyl.StateVector(6)
costs = crocoddyl.CostModelSum(state, 1)

weights = np.array([0.,0.,0.,10.,10.,10.])
# xRef = np.zeros(6)
xRef = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
costs.addCost("comTracking", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(weights), xRef, 1), 1e3)
costs.addCost("uReg", crocoddyl.CostModelControl(state, 1), 1e-3) ## ||u||^2

model = DifferentialActionModelVariableHeightPendulum(costs)
data = model.createData()
dt = 1e-3
num_nodes = 30
m = crocoddyl.IntegratedActionModelEuler(model, dt)
x_init = np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0])
# x_init = np.zeros(6)
problem = crocoddyl.ShootingProblem(x_init, [m]*num_nodes, m)
solver = crocoddyl.SolverFDDP(problem)
log = crocoddyl.CallbackLogger()
solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
t0 = time.time()
solver.solve()
print 'Time of iteration consumed', time.time()-t0

crocoddyl.plotOCSolution(log.xs, log.us)
crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)





