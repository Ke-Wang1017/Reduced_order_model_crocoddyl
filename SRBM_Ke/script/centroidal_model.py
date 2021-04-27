import crocoddyl
import numpy as np
import time
import math

class DifferentialActionModelCentroidal(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs, m, Ig):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), 4) # nu = 1
        self.uNone = np.zeros(self.nu)

        self.m = m
        self.Ig = Ig
        self.g = 9.81
        self.costs = costs
        self.u_lb = np.array([100., -0.8, -0.12, 0.0])
        self.u_ub = np.array([2.0*self.m*self.g, 0.8, 0.12, 0.05])


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

        data.Fu[:, :] = np.array([[(c_x-u_x)/(self.m*(c_z-u_z)), -f_z/((c_z-u_z)*self.m), 0., self.m*f_z*(c_x-u_x)/((c_z-u_z)*self.m)**2],
                                 [(c_y-u_y)/(self.m*(c_z-u_z)), 0., -f_z /((c_z - u_z) * self.m), self.m * f_z * (c_y - u_y) / ((c_z - u_z) * self.m) ** 2],
                                 [1.0/self.m, 0., 0., 0.]]) # needs to be modified
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataCentroidal(self)


class DifferentialActionDataCentroidal(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        shared_data = crocoddyl.DataCollectorAbstract()
        self.costs = model.costs.createData(shared_data)
        self.costs.shareMemory(self)


def createPhaseModel(cop, xref=np.array([0.0, 0.0, 0.86, 0.1, 0.0, 0.0]), Wx=np.array([0., 0., 10., 10., 10., 10.]), Wu=np.array([0., 50., 50., 1.]), wxreg=1, wureg=5, wxbox=1, dt=2e-2):
    state = crocoddyl.StateVector(6)
    runningCosts = crocoddyl.CostModelSum(state, 4)
    uRef = np.hstack([np.zeros(1), cop])
    xRef = xref
    ub = np.hstack([cop, np.zeros(3)]) + np.array([0.3, 0.055, 0.95, 7., 7., 3])
    lb = np.hstack([cop, np.zeros(3)]) + np.array([-0.3, -0.055, 0.75, -7., -7., -3])
    runningCosts.addCost("comBox", crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)), xRef, 4), wxbox)
    runningCosts.addCost("comReg", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(Wx), xRef, 4), wxreg)
    runningCosts.addCost("uTrack", crocoddyl.CostModelControl(state, crocoddyl.ActivationModelWeightedQuad(Wu), uRef), wureg) ## ||u||^2
    runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, 4), wureg*10) ## ||u||^2
    model = DifferentialActionModelCentroidal(runningCosts, m=95.0)
    return crocoddyl.IntegratedActionModelEuler(model, dt)

def createTerminalModel(cop):
    return createPhaseModel(cop, xref=np.array([0.0, 0.0, 0.86, 0.02, 0.0, 0.0]), Wx=np.array([0., 0., 100., 30., 30., 150.]), wxreg=1e6, dt=0.)

m1 = createPhaseModel(np.array([0.0, 0.0, 0.00]))
m2 = createPhaseModel(np.array([0.0, -0.08, 0.00]))
m3 = createPhaseModel(np.array([0.1, 0.0, 0.00]))
m4 = createPhaseModel(np.array([0.2, 0.08, 0.00]))
m5 = createPhaseModel(np.array([0.2, 0.0, 0.00]))
mT = createTerminalModel(np.array([0.2, 0.0, 0.00]))

num_nodes_single_support = 50
num_nodes_double_support = 20
locoModel = [m1]*num_nodes_double_support
locoModel += [m2]*num_nodes_single_support
locoModel += [m3]*num_nodes_double_support
locoModel += [m4]*num_nodes_single_support
locoModel += [m5]*num_nodes_double_support

x_init = np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0])
# x_init = np.zeros(6)
problem = crocoddyl.ShootingProblem(x_init, locoModel, mT)
solver = crocoddyl.SolverBoxFDDP(problem)
# solver = crocoddyl.SolverFDDP(problem)
log = crocoddyl.CallbackLogger()
solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
u_init = np.array([931.95, 0.0, 0.0, 0.001])
t0 = time.time()
# u_init = [m.quasiStatic(d, x_init) for m,d in zip(problem.runningModels, problem.runningDatas)]
solver.solve([x_init]*(problem.T + 1), [u_init]*problem.T, 100) # x init, u init, max iteration
# solver.solve()  # x init, u init, max iteration

print('Time of iteration consumed', time.time()-t0)

crocoddyl.plotOCSolution(log.xs[:], log.us)
# crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)


def plotComMotion(xs, us):
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cx = [x[0] for x in xs]
    cy = [x[1] for x in xs]
    cz = [x[2] for x in xs]
    cxdot = [x[3] for x in xs]
    cydot = [x[4] for x in xs]
    czdot = [x[5] for x in xs]
    f_z = [u[0] for u in us]
    u_x = [u[1] for u in us]
    u_y = [u[2] for u in us]
    u_z = [u[3] for u in us]

    plt.plot(cx)
    plt.show()
    plt.plot(cy)
    plt.show()
    plt.plot(cz)
    plt.show()
    plt.plot(u_x)
    plt.show()
    plt.plot(u_y)
    plt.show()


plotComMotion(solver.xs, solver.us)

# solver.
import trajectory_publisher

