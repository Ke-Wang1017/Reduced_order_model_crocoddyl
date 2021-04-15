import crocoddyl
import numpy as np
import time
import math

class DifferentialActionModelVariableHeightPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), 1) # nu = 1
        self.uNone = np.zeros(self.nu)

        self.m = 95.0
        self.g = 9.81
        self.foot_location = np.array([0.0, 0.0, 0.000001])
        self.costs = costs
        self.u_lb = np.array([100.])
        self.u_ub = np.array([5.0*self.m*self.g])
        self.state.lb = np.array([-1e6, -1e6, 0.3, -1e6, -1e6, -self.g])
        self.state.ub = np.array([1e6, 1e6, 1.3, 20, 20, 4*self.g])


    def setFootLocation(self, footLocation):
        self.foot_location = footLocation

    def calc(self, data, x, u):
        if u is None:
            u = self.uNone
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x # how do you know cdot is the diff of c? From the Euler integration model
        u_x, u_y, u_z = self.foot_location
        f_z = u.item(0)

        cdotdot_x = f_z * (c_x-u_x) / ((c_z-u_z) * self.m)
        cdotdot_y = f_z * (c_y-u_y) / ((c_z-u_z) * self.m)
        cdotdot_z = f_z/self.m - self.g
        data.xout = np.array([cdotdot_x, cdotdot_y, cdotdot_z]).T

        # compute the cost residual
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

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
runningCosts = crocoddyl.CostModelSum(state, 1)
terminalCosts = crocoddyl.CostModelSum(state, 1)

weights = np.array([0., 0., 10., 0., 50., 0.])
weights_terminal = np.array([0., 0., 100., 0., 5., 0.])
# xRef = np.zeros(6)
xRef = np.array([0.0, 0.0, 0.98, 0.0, 0.0, 0.0])
runningCosts.addCost("comTracking", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(weights), xRef, 1), 1e3)
runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, 1), 1e-3) ## ||u||^2
terminalCosts.addCost("comTracking", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(weights_terminal), xRef, 1), 1e5)
model1 = DifferentialActionModelVariableHeightPendulum(runningCosts)
model2 = DifferentialActionModelVariableHeightPendulum(runningCosts)
model3 = DifferentialActionModelVariableHeightPendulum(runningCosts)
model4 = DifferentialActionModelVariableHeightPendulum(runningCosts)
# model5 = DifferentialActionModelVariableHeightPendulum(runningCosts)
modelT = DifferentialActionModelVariableHeightPendulum(terminalCosts)
model1.setFootLocation(np.array([0.0, -0.075, 0.001]))
model2.setFootLocation(np.array([0.0, 0.0, 0.001]))
model3.setFootLocation(np.array([0.0, 0.075, 0.001]))
model4.setFootLocation(np.array([0.0, 0.0, 0.001]))
# data = model.createData() # seems not needed for this
dt = 1e-2
num_nodes_single_support = 80
num_nodes_double_support = 40
m1 = crocoddyl.IntegratedActionModelEuler(model1, dt)
m2 = crocoddyl.IntegratedActionModelEuler(model2, dt)
m3 = crocoddyl.IntegratedActionModelEuler(model3, dt)
m4 = crocoddyl.IntegratedActionModelEuler(model4, dt)
mT = crocoddyl.IntegratedActionModelEuler(modelT)

locoModel = [m1]*num_nodes_single_support
locoModel += [m2]*num_nodes_double_support
locoModel += [m3]*num_nodes_single_support
locoModel += [m4]*num_nodes_double_support
# locoModel += [m5]*num_nodes_double_support

x_init = np.array([0.0, 0.0, 0.981, 0.0, 0.0, 0.0])
# x_init = np.zeros(6)
problem = crocoddyl.ShootingProblem(x_init, locoModel, mT)
solver = crocoddyl.SolverBoxFDDP(problem)
# solver = crocoddyl.SolverFDDP(problem)
log = crocoddyl.CallbackLogger()
solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
t0 = time.time()
solver.solve()

print 'Time of iteration consumed', time.time()-t0

crocoddyl.plotOCSolution(log.xs[:], log.us)
# crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)


def plotComMotion(xs):
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cx = [x[0] for x in xs]
    cy = [x[1] for x in xs]
    cz = [x[2] for x in xs]
    cxdot = [x[3] for x in xs]
    cydot = [x[4] for x in xs]
    czdot = [x[5] for x in xs]
    plt.plot(cy)
    plt.show()
    plt.plot(cz)
    plt.show()

plotComMotion(solver.xs)
