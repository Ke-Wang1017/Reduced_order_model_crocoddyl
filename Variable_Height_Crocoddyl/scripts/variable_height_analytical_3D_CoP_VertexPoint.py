import numpy as np
import time
from scipy.interpolate import interp1d
import crocoddyl
import rospy
import matplotlib.pyplot as plt
import example_robot_data
import pinocchio

class DifferentialActionModelVariableHeightPendulum(
        crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, 4)  # nu = 1

        self._m = pinocchio.computeTotalMass(self.state.pinocchio)
        self._g = abs(self.state.pinocchio.gravity.linear[2])
        self.contacts = crocoddyl.ContactModelMultiple(state, 4)
        self.contacts.addContact(
            "single",
            crocoddyl.ContactModel3D(
                state, crocoddyl.FrameTranslation(0, np.zeros(3)), 4))
        self.costs = costs
        self.u_lb = np.array([100., -0.8, -0.12, 0.0])
        self.u_ub = np.array([2.0 * self._m * self._g, 0.8, 0.12, 0.05])

    def calc(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x  # how do you know cdot is the diff of c? From the Euler integration model
        f_z, u_x, u_y, u_z = u.item(0), u.item(1), u.item(2), u.item(3)

        cdotdot_x = f_z * (c_x - u_x) / ((c_z - u_z) * self._m)
        cdotdot_y = f_z * (c_y - u_y) / ((c_z - u_z) * self._m)
        cdotdot_z = f_z / self._m - self._g
        data.xout = np.array([cdotdot_x, cdotdot_y, cdotdot_z]).T

        # compute the cost residual
        data.contacts.contacts["single"].f.linear = np.array([
            f_z * (c_x - u_x) / (c_z - u_z), f_z * (c_y - u_y) / (c_z - u_z),
            f_z
        ])
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        f_z, u_x, u_y, u_z = u.item(0), u.item(1), u.item(2), u.item(3)

        data.Fx[:, :] = np.array(
            [[
                f_z / ((c_z - u_z) * self._m), 0.0,
                -f_z * (c_x - u_x) / ((c_z - u_z)**2 * self._m), 0.0, 0.0, 0.0
            ],
             [
                 0.0, f_z / ((c_z - u_z) * self._m),
                 -f_z * (c_y - u_y) / ((c_z - u_z)**2 * self._m), 0.0, 0.0, 0.0
             ], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        data.Fu[:, :] = np.array(
            [[(c_x - u_x) / (self._m * (c_z - u_z)),
              -f_z / ((c_z - u_z) * self._m), 0.,
              self._m * f_z * (c_x - u_x) / ((c_z - u_z) * self._m)**2],
             [(c_y - u_y) / (self._m * (c_z - u_z)), 0.,
              -f_z / ((c_z - u_z) * self._m),
              self._m * f_z * (c_y - u_y) / ((c_z - u_z) * self._m)**2],
             [1.0 / self._m, 0., 0., 0.]])  # needs to be modified

        data.contacts.contacts["single"].df_dx[:, :] = \
            np.array([[f_z / (c_z - u_z), 0.0, -f_z * (c_x - u_x) / ((c_z - u_z)**2), 0.0, 0.0, 0.0],
                      [0.0, f_z / (c_z - u_z), -f_z * (c_y - u_y) / ((c_z - u_z)**2), 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        data.contacts.contacts["single"].df_du[:, :] = np.array([[(c_x - u_x) / (c_z - u_z), -f_z / (c_z - u_z), 0.0, f_z * (c_x - u_x) / ((c_z - u_z)**2)],
                                                                 [(c_y - u_y) / (c_z - u_z), 0.0, -f_z / (c_z - u_z), f_z * (c_y - u_y) / ((c_z - u_z)**2)],
                                                                 [1.0, 0.0, 0.0, 0.0]])

        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataVariableHeightPendulum(self)


class DifferentialActionDataVariableHeightPendulum(
        crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = model.state.pinocchio.createData()
        self.contacts = model.contacts.createData(self.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibodyInContact(
            self.pinocchio, self.contacts)
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)

def buildSRBMFromRobot(robot_model):
    model = pinocchio.Model()
    placement = pinocchio.SE3.Identity()
    joint_com = model.addJoint(0, pinocchio.JointModelTranslation(), placement, 'com') # for inverted pendulum model
    # joint_com = model.addJoint(0, pinocchio.JointModelFreeFlyer(), placement, 'com') # for single rigid body model
    body_inertia = pinocchio.Inertia.Zero()
    body_inertia.mass = pinocchio.computeTotalMass(robot_model.model)
    data = robot_model.data
    q0 = robot_model.q0
    com = robot_model.com(q0)
    body_placement = placement
    body_placement.translation[0] = com[0]
    body_placement.translation[1] = com[1]
    body_placement.translation[2] = com[2]
    model.appendBodyToJoint(joint_com, body_inertia, body_placement)

    return model


def createPhaseModel(robot_model,
                     cop,
                     xref=np.array([0.0, 0.0, 0.86, 0.15, 0.0, 0.0]),
                     nsurf=np.array([0., 0., 1.]).T,
                     mu=0.7,
                     Wx=np.array([0., 0., 10., 10., 10., 10.]),
                     Wu=np.array([0., 50., 50., 1.]),
                     wxreg=1,
                     wureg=5,
                     wutrack=50,
                     wxbox=1,
                     dt=2e-2):
    state = crocoddyl.StateVector(6)
    model = buildSRBMFromRobot(robot_model)
    multibody_state = crocoddyl.StateMultibody(model)
    runningCosts = crocoddyl.CostModelSum(state, 4)
    uRef = np.hstack([np.zeros(1), cop])
    xRef = xref
    nSurf = nsurf
    Mu = mu
    # cone = crocoddyl.FrictionCone(nSurf, Mu, 1, False)
    ub = np.hstack([cop, np.zeros(3)]) + np.array(
        [0.3, 0.055, 0.95, 7., 7., 3])
    lb = np.hstack([cop, np.zeros(3)]) + np.array(
        [-0.3, -0.055, 0.75, -7., -7., -3])
    runningCosts.addCost(
        "comBox",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(lb, ub)),
            crocoddyl.ResidualModelState(state, xRef, 4)), wxbox)
    runningCosts.addCost(
        "comReg",
        crocoddyl.CostModelResidual(
            state, crocoddyl.ActivationModelWeightedQuad(Wx),
            crocoddyl.ResidualModelState(state, xRef, 4)), wxreg)
    runningCosts.addCost("uTrack",
                         crocoddyl.CostModelResidual(
                             state, crocoddyl.ActivationModelWeightedQuad(Wu),
                             crocoddyl.ResidualModelControl(state, uRef)),
                         wureg)  ## ||u||^2
    runningCosts.addCost("uReg",
                         crocoddyl.CostModelResidual(
                             state, crocoddyl.ResidualModelControl(state, 4)),
                         wutrack)  ## ||u||^2
    cone = crocoddyl.FrictionCone(np.eye(3), Mu)
    runningCosts.addCost(
        "frictionPenalization",
        crocoddyl.CostModelResidual(
            multibody_state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)),
            crocoddyl.ResidualModelContactFrictionCone(multibody_state, 0,
                                                       cone, 4)), 1e2)
    model = DifferentialActionModelVariableHeightPendulum(
        multibody_state, runningCosts)
    return crocoddyl.IntegratedActionModelEuler(model, dt)


def createTerminalModel(robot_model, cop):
    return createPhaseModel(robot_model,
                            cop,
                            xref=np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0]),
                            Wx=np.array([0., 10., 100., 10., 10., 150.]),
                            wxreg=1e6,
                            dt=0.)


def plotComMotion(xs, us):
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


if __name__ == "__main__":

    foot_holds = np.array([[0.0, 0.0, 0.0], [0.0, -0.085,
                                             0.0], [0.05, 0.0, 0.0],
                           [0.1, 0.085, 0.0], [0.1, 0.0, 0.0]])
    phase = np.array([0, -1, 0, 1, 0])  # 0: double, 1: left, -1: right
    len_steps = phase.shape[0]
    robot_model = example_robot_data.load("talos")
    foot_placements = np.zeros((len_steps, 6))
    foot_orientations = np.zeros((len_steps, 6))
    for i in range(len_steps):
        if phase[i] == 0:
            if i == 0:
                foot_placements[i, 0] = foot_holds[i, 0]
                foot_placements[i, 2] = foot_holds[i, 2]
                foot_placements[i, 3] = foot_holds[i, 0]
                foot_placements[i, 5] = foot_holds[i, 2]
                foot_placements[i, 1] = foot_holds[i, 1] + 0.085
                foot_placements[i, 4] = foot_holds[i, 1] - 0.085
            else:
                foot_placements[i, :] = foot_placements[i - 1, :]
        elif phase[i] == -1:
            foot_placements[i, 3:] = foot_holds[i, :]
            foot_placements[i, :3] = foot_placements[i - 1, :3]
        elif phase[i] == 1:
            foot_placements[i, :3] = foot_holds[i, :]
            foot_placements[i, 3:] = foot_placements[i - 1, 3:]
    print('Foot placements are ', foot_placements)
    num_nodes_single_support = 40
    num_nodes_double_support = 20
    dt = 2e-2
    support_indexes = np.array([0, -1, 0, 1, 0])
    support_durations = np.zeros(len_steps)
    for i in range(len_steps):
        if support_indexes[i] == 0:
            support_durations[i] = num_nodes_double_support * dt
        else:
            support_durations[i] = num_nodes_single_support * dt

    locoModel = [createPhaseModel(robot_model, foot_holds[0, :])]
    cop = np.zeros(3)
    for i in range(len(phase)):
        if phase[i] == 0:
            if i == 0:
                for j in range(num_nodes_double_support):
                    tmp_cop = foot_holds[0, :] + (
                        foot_holds[1, :] -
                        foot_holds[0, :]) * (j + 1) / num_nodes_double_support
                    cop = np.vstack((cop, tmp_cop))
                    tmp_model = createPhaseModel(robot_model,
                                                 tmp_cop,
                                                 xref=np.array([
                                                     0.0, 0.0,
                                                     0.86 + tmp_cop[2], 0.1,
                                                     0.0, 0.0
                                                 ]))
                    locoModel += [tmp_model]
            elif i == len(phase) - 1:
                for j in range(num_nodes_double_support):
                    tmp_cop = foot_holds[i - 1, :] + (
                        foot_holds[i, :] - foot_holds[i - 1, :]) * (
                            j + 1) / num_nodes_double_support
                    cop = np.vstack((cop, tmp_cop))
                    tmp_model = createPhaseModel(robot_model,
                                                 tmp_cop,
                                                 xref=np.array([
                                                     0.0, 0.0,
                                                     0.86 + tmp_cop[2], 0.1,
                                                     0.0, 0.0
                                                 ]))
                    locoModel += [tmp_model]
            else:
                for j in range(num_nodes_double_support):
                    tmp_cop = foot_holds[i - 1, :] + (
                        foot_holds[i + 1, :] - foot_holds[i - 1, :]) * (
                            j + 1) / num_nodes_double_support
                    cop = np.vstack((cop, tmp_cop))
                    tmp_model = createPhaseModel(robot_model,
                                                 tmp_cop,
                                                 xref=np.array([
                                                     0.0, 0.0,
                                                     0.86 + tmp_cop[2], 0.1,
                                                     0.0, 0.0
                                                 ]))
                    locoModel += [tmp_model]

        if phase[i] == 1 or phase[i] == -1:
            locoModel += [
                createPhaseModel(robot_model,
                                 foot_holds[i, :],
                                 xref=np.array([
                                     0.0, 0.0, 0.86 + tmp_cop[2], 0.1, 0.0, 0.0
                                 ]))
            ] * num_nodes_single_support
            for j in range(num_nodes_single_support):
                cop = np.vstack((cop, foot_holds[i, :]))
    mT = createTerminalModel(robot_model, np.array([0.08, 0.0, 0.00]))

    # import matplotlib.pyplot as plt
    # plt.plot(cop[:,1])
    # plt.show()

    u_init = np.array([931.95, 0.0, 0.0, 0.001])
    x_init = np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0])
    problem = crocoddyl.ShootingProblem(x_init, locoModel, mT)
    solver = crocoddyl.SolverBoxFDDP(problem)
    log = crocoddyl.CallbackLogger()
    solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
    t0 = time.time()
    solver.solve([x_init] * (problem.T + 1), [u_init] * problem.T,
                 10)  # x init, u init, max iteration
    # solver.solve()  # x init, u init, max iteration

    print('Time of iteration consumed', time.time() - t0)
    swing_height = 0.10
    swing_height_offset = 0.01
    com_gain = [30.0, 7.5, 1.0]

    com_pos = np.array([x[:3] for x in solver.xs])
    com_vel = np.array([x[3:] for x in solver.xs])
    com_acc = np.vstack((np.diff(com_vel, axis=0) / dt, np.zeros(3)))
    total_time = np.cumsum(support_durations)[-1]
    time = np.linspace(0, total_time, com_pos.shape[0])
    # sampling based on the frequency
    f_pos_x = interp1d(time,
                       com_pos[:, 0],
                       fill_value=(com_pos[0, 0], com_pos[-1, 0]),
                       bounds_error=False)
    f_pos_y = interp1d(time,
                       com_pos[:, 1],
                       fill_value=(com_pos[0, 1], com_pos[-1, 1]),
                       bounds_error=False)
    f_pos_z = interp1d(time,
                       com_pos[:, 2],
                       fill_value=(com_pos[0, 2], com_pos[-1, 2]),
                       bounds_error=False)
    f_vel_x = interp1d(time,
                       com_vel[:, 0],
                       fill_value=(com_vel[0, 0], com_vel[-1, 0]),
                       bounds_error=False)
    f_vel_y = interp1d(time,
                       com_vel[:, 1],
                       fill_value=(com_vel[0, 1], com_vel[-1, 1]),
                       bounds_error=False)
    f_vel_z = interp1d(time,
                       com_vel[:, 2],
                       fill_value=(com_vel[0, 2], com_vel[-1, 2]),
                       bounds_error=False)
    f_acc_x = interp1d(time,
                       com_acc[:, 0],
                       fill_value=(com_acc[0, 0], com_acc[-1, 0]),
                       bounds_error=False)
    f_acc_y = interp1d(time,
                       com_acc[:, 1],
                       fill_value=(com_acc[0, 1], com_acc[-1, 1]),
                       bounds_error=False)
    f_acc_z = interp1d(time,
                       com_acc[:, 2],
                       fill_value=(com_acc[0, 2], com_acc[-1, 2]),
                       bounds_error=False)
    freq = 500
    sample_num = int(freq * total_time)
    time_sample = np.linspace(0, total_time, sample_num)

    com_traj_sample = np.vstack(
        (time_sample, f_pos_x(time_sample), f_pos_y(time_sample),
         f_pos_z(time_sample), f_vel_x(time_sample), f_vel_y(time_sample),
         f_vel_z(time_sample), f_acc_x(time_sample), f_acc_y(time_sample),
         f_acc_z(time_sample)))

    plotComMotion(solver.xs, solver.us)
    # plt.plot(com_traj_sample[7, :])
    # plt.show()

    import trajectory_publisher

    rospy.init_node('listener', anonymous=True)

    trajectory_publisher.publish_all(com_traj_sample.T, support_durations,
                                     support_indexes, foot_placements,
                                     foot_orientations, swing_height,
                                     swing_height_offset, com_gain)

    # np.savez('crocoddyl_data.npz', com_traj=com_traj_sample, support_durations=support_durations, support_indexes=support_indexes,
    #         foot_placements=foot_placements, foot_orientations=foot_orientations)
    # crocoddyl.plotOCSolution(log.xs[:], log.us)
    # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)

    # np.save('../foot_holds.npy', np.concatenate((foot_holds, np.array([phase]).T), axis=1))
    # np.save('../com_traj.npy', solver.xs)
    # np.save('../control_traj.npy', solver.us)

    # solver.
