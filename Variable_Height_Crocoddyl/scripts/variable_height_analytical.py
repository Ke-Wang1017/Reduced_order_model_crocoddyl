import crocoddyl
import numpy as np
import time
import math

class DifferentialActionModelVariableHeightPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), 4) # nu = 1
        self.uNone = np.zeros(self.nu)

        self.m = 95.0
        self.g = 9.81
        self.foot_location = np.array([0.0, 0.0, 0.000001])
        self.costs = costs
        self.u_lb = np.array([100., -0.05, -0.12, 0.0])
        self.u_ub = np.array([3.0*self.m*self.g, 0.05, 0.12, 0.2])


    def setFootLocation(self, footLocation):
        self.foot_location = footLocation

    def calc(self, data, x, u):
        if u is None:
            u = self.uNone
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x # how do you know cdot is the diff of c? From the Euler integration model
        f_z, u_x, u_y, u_z = u.item(0), u.item(1), u.item(2), u.item(3)

        cdotdot_x = f_z * (c_x-u_x) / ((c_z-u_z) * self.m)
        cdotdot_y = f_z * (c_y-u_y) / ((c_z-u_z) * self.m)
        cdotdot_z = f_z/self.m - self.g
        data.xout = np.array([cdotdot_x, cdotdot_y, cdotdot_z]).T

        # compute the cost residual
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        f_z, u_x, u_y, u_z = u.item(0), u.item(1), u.item(2), u.item(3)

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


def createPhaseModel(cop, Wx=np.array([0., 0., 20., 0., 0., 10.]), wxreg=5e-1, wureg=1e-1, wxbox=1e5, dt=2e-2):
    state = crocoddyl.StateVector(6)
    runningCosts = crocoddyl.CostModelSum(state, 1)
    xRef = np.array([0.0, 0.0, 0.98, 0.0, 0.0, 0.0])
    ub = np.hstack([cop, np.zeros(3)]) + np.array([0.5, 0.1, 1.2, 5., 5., 5])
    lb = np.hstack([cop, np.zeros(3)]) + np.array([-0.5, -0.1, 0.7, -5., -5., -5])
    runningCosts.addCost("comBox", crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)), xRef, 1), wxbox)
    runningCosts.addCost("comReg", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(Wx), xRef, 1), wxreg)
    runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, 1), wureg) ## ||u||^2
    model = DifferentialActionModelVariableHeightPendulum(runningCosts)
    model.setFootLocation(cop)
    return crocoddyl.IntegratedActionModelEuler(model, dt)

def createTerminalModel(cop):
    return createPhaseModel(cop, np.array([0., 0., 20., 0., 0., 10.]), wxreg=1e2, dt=0.)

m1 = createPhaseModel(np.array([0.0, -0.08, 0.001]))
m2 = createPhaseModel(np.array([0.0, 0.0, 0.001]))
m3 = createPhaseModel(np.array([0.0, 0.08, 0.001]))
m4 = createPhaseModel(np.array([0.0, 0.0, 0.001]))
mT = createTerminalModel(np.array([0.0, 0.0, 0.001]))
# data = model.createData() # seems not needed for this

num_nodes_single_support = 40
num_nodes_double_support = 20
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
u_init = [m.quasiStatic(d, x_init) for m,d in zip(problem.runningModels, problem.runningDatas)]
# solver.solve([x_init]*(problem.T + 1), u_init, 100) # x init, u init, max iteration
solver.solve()  # x init, u init, max iteration

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
    plt.plot(czdot)
    plt.show()

plotComMotion(solver.xs)
