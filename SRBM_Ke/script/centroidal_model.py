import crocoddyl
import numpy as np
import time
import math
import utils

class DifferentialActionModelCentroidal(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, costs, m):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(12), 12) # nu = 12
        self.uNone = np.zeros(self.nu)

        self.m = m
        self.Ig = np.diag([0.00578574, 0.01938108, 0.02476124]) # can be get from URDF

        self.g = np.zeros(6)
        self.g[2] = -9.81
        self.costs = costs
        self.footholds = np.array(
            [[0.19, 0.19, -0.19, -0.19],
             [0.15005, -0.15005, 0.15005, -0.15005],
             [0.0, 0.0, 0.0, 0.0]])
        #Normal vector for friction cone
        self.nsurf = np.array([0., 0., 1.]).T # flat ground
        self.S = np.ones(4)

        # self.u_lb = np.array([100., -0.8, -0.12, 0.0])
        # self.u_ub = np.array([2.0*self.m*self.g, 0.8, 0.12, 0.05])


    def calc(self, data, x, u):
        # Levers Arms used in B, can I put lever-arms into data???
        data.L = self.footholds - np.array(x[:3]).transpose() # broadcast x
        H = utils.euler_matrix(x[3],x[4],x[5])
        R = H[:3,:3]
        data.I_inv = np.linalg.inv(np.dot(R, self.gI))
        for i in range(4):
            # if feet in touch with ground
            if self.S[i] != 0:
                data.B[:3, (i*3):((i+1)*3)] = np.identity(3)/self.m #
                data.B[-3:, (i*3):((i+1)*3)] = np.dot(data.I_inv, utils.getSkew(data.L[:, i])) # another term needs added
        # Compute friction cone
        # self.costFriction(u)
        data.xout = np.dot(data.B,u) + self.g

        # compute the cost residual
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        for i in range(4):
            if self.S[i] != 0:
                data.derivative_B[-3:, 0] = - np.dot(data.I_inv, np.cross([1, 0, 0], [u[3 * i], u[3 * i + 1],
                                                                                                u[3 * i + 2]]))  # \x
                data.derivative_B[-3:, 1] = - np.dot(data.I_inv, np.cross([0, 1, 0], [u[3 * i], u[3 * i + 1],
                                                                                                u[3 * i + 2]]))  # \y
                data.derivative_B[-3:, 2] = - np.dot(data.I_inv, np.cross([0, 0, 1], [u[3 * i], u[3 * i + 1],
                                                                                                u[3 * i + 2]]))  # \z
        data.Fx[:,:] = data.derivative_B[:,:]
        data.Fu[:,:] = data.B[:,:]
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialActionDataCentroidal(self)
        data.B = np.zeros((6, 12))
        data.L = np.zeros((3, 4)) # 4 contact points
        data.derivative_B = np.zeros((6, 12))
        data.I_inv = np.identity(3)
        return data

    def updateModel(self, foothold, nsurf, contact_selection, Ig):
        self.footholds = foothold
        self.nsurf = nsurf
        self.S = contact_selection
        self.Ig = Ig

class DifferentialActionDataCentroidal(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        shared_data = crocoddyl.DataCollectorAbstract()
        self.costs = model.costs.createData(shared_data)
        self.costs.shareMemory(self)


def createPhaseModel(foothold, nsurf, mu, xref=np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
                     Wx=np.ones(12), Wu=np.ones(12), wxreg=1, wureg=5, wufriction=5, wxbox=1, dt=2e-2):
    state = crocoddyl.StateVector(12)
    runningCosts = crocoddyl.CostModelSum(state, 12)
    Foothold = foothold
    Nsurf = nsurf
    Mu = mu
    xRef = xref
    cone = crocoddyl.FrictionCone(Nsurf, Mu, 4, False)
    x_ub = np.hstack([Foothold, np.zeros(3)]) + np.array([0.3, 0.055, 0.95, 7., 7., 3]) # can be modified
    x_lb = np.hstack([Foothold, np.zeros(3)]) + np.array([-0.3, -0.055, 0.75, -7., -7., -3])
    u_lb = np.tile(cone.lb, 4)  # force based on friction cone
    u_ub = np.tile(cone.ub, 4)
    runningCosts.addCost("comBox", crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub)), xRef, 4), wxbox)
    runningCosts.addCost("comReg", crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(Wx), xRef, 12), wxreg)
    runningCosts.addCost("frictionCone", crocoddyl.CostModelControl(state, crocoddyl.ActivationModelQuadraticBarrier((crocoddyl.ActivationBounds(u_lb, u_ub)))), wufriction)
    runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, 12), wureg*10) ## ||u||^2
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

