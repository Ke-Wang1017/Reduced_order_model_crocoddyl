import crocoddyl
import numpy as np
import time
import math


class DifferentialActionModelLinearInvertedPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(4), 2)  # nu = 1
        self.uNone = np.zeros(self.nu)

        self.m = 95.0
        self.g = 9.81
        self.xz = 0.86
        self.ref_foot_location = np.array([0.0, 0.0, 0.000001])
        self.costs = costs
        self.u_lb = np.array([-0.05, -0.1])
        self.u_ub = np.array([1.0, 0.1])

    def setRefFootLocation(self, footLocation):
        self.ref_foot_location = footLocation

    def calc(self, data, x, u):
        if u is None:
            u = self.uNone
        c_x, c_y, cdot_x, cdot_y = x  # how do you know cdot is the diff of c? From the Euler integration model
        u_x, u_y = u.item(0), u.item(1)

        cdotdot_x = self.g * (c_x - u_x) / self.xz
        cdotdot_y = self.g * (c_y - u_y) / self.xz
        data.xout = np.array([cdotdot_x, cdotdot_y]).T

        # compute the cost residual
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        c_x, c_y, cdot_x, cdot_y = x
        u_x, u_y = u[0].item(), u[1].item()

        data.Fx[:, :] = np.array([[self.g / self.xz, 0.0, 0.0, 0.0], [0.0, self.g / self.xz, 0.0, 0.0]])
        data.Fu[:] = np.array([[-self.g / self.xz, 0.0], [0.0, -self.g / self.xz]])
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataLinearInvertedPendulum(self)


class DifferentialActionDataLinearInvertedPendulum(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        shared_data = crocoddyl.DataCollectorAbstract()
        self.costs = model.costs.createData(shared_data)
        self.costs.shareMemory(self)


def createPhaseModel(cop,
                     Wx=np.array([0., 0., 5., 10.]),
                     Wu=np.array([10., 20.]),
                     wxreg=1e-1,
                     wureg=5,
                     wxbox=1e5,
                     dt=2e-2):
    state = crocoddyl.StateVector(4)
    runningCosts = crocoddyl.CostModelSum(state, 2)
    xRef = np.array([0.0, 0.0, 0.1, 0.0])
    uRef = cop
    ub = np.hstack([cop, np.zeros(2)]) + np.array([0.5, 0.1, 5., 5.])
    lb = np.hstack([cop, np.zeros(2)]) + np.array([-0.5, -0.1, -5., -5.])
    runningCosts.addCost(
        "comBox",
        crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
                                 xRef, 2), wxbox)
    runningCosts.addCost("comReg", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(Wx), xRef, 2),
                         wxreg)
    runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, crocoddyl.ActivationModelWeightedQuad(Wu), uRef),
                         wureg)  ## ||u||^2
    model = DifferentialActionModelLinearInvertedPendulum(runningCosts)
    return crocoddyl.IntegratedActionModelEuler(model, dt)


def createTerminalModel(cop):
    xRef = np.array([0.0, 0.0, 0.0, 0.0])
    return createPhaseModel(cop, np.array([0., 1., 5., 5.]), wxreg=1e1, dt=0.)


m1 = createPhaseModel(np.array([0.0, 0.0]))
m2 = createPhaseModel(np.array([0.1, -0.08]))
m3 = createPhaseModel(np.array([0.12, 0.0]))
m4 = createPhaseModel(np.array([0.2, 0.08]))
m5 = createPhaseModel(np.array([0.22, 0.0]))
mT = createTerminalModel(np.array([0.3, 0.0]))
# data = model.createData() # seems not needed for this

num_nodes_single_support = 40
num_nodes_double_support = 20
locoModel = [m1] * num_nodes_double_support
locoModel += [m2] * num_nodes_single_support
locoModel += [m3] * num_nodes_double_support
locoModel += [m4] * num_nodes_single_support
locoModel += [m5] * num_nodes_double_support

x_init = np.array([0.0, 0.0, 0.0, 0.0])
# x_init = np.zeros(6)
problem = crocoddyl.ShootingProblem(x_init, locoModel, mT)
solver = crocoddyl.SolverBoxFDDP(problem)
# solver = crocoddyl.SolverFDDP(problem)
log = crocoddyl.CallbackLogger()
solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
t0 = time.time()
# u_init = [m.quasiStatic(d, x_init) for m,d in zip(problem.runningModels, problem.runningDatas)]
solver.solve()  # x init, u init, max iteration
# solver.solve([x_init]*(problem.T + 1), u_init, 100) # x init, u init, max iteration

print 'Time of iteration consumed', time.time() - t0

crocoddyl.plotOCSolution(log.xs[:], log.us)
# crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)


def plotComMotion(xs):
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cx = [x[0] for x in xs]
    cy = [x[1] for x in xs]
    cxdot = [x[2] for x in xs]
    cydot = [x[3] for x in xs]
    plt.plot(cx)
    plt.show()
    plt.plot(cy)
    plt.show()
    plt.plot(cydot)
    plt.show()


plotComMotion(solver.xs)
