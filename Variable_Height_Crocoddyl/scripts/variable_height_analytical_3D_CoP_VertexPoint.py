import time

import crocoddyl
import example_robot_data
import matplotlib.pyplot as plt
import numpy as np
import pinocchio

from util import rotx, roty, rotz
from ResidualControlClass import ControlResidual


class DifferentialActionModelVariableHeightPendulum(
    crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costs):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, 8)  # nu = 9: f_z, vertex points(8)

        self._m = pinocchio.computeTotalMass(self.state.pinocchio)
        self._g = abs(self.state.pinocchio.gravity.linear[2])
        self.contacts = crocoddyl.ContactModelMultiple(state, 8)
        self.contacts.addContact(
            "single",
            crocoddyl.ContactModel3D(
                state, crocoddyl.FrameTranslation(0, np.zeros(3)), 8))
        self.costs = costs
        foot_size = [0.2, 0.1, 0]
        self.Vs = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1],
                            [1, -1, 1]]) * foot_size / 2  # distance of 4 vertexes from the center of foot
        self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([2.0 * self._m * self._g, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.foot_pos = np.zeros(6)
        self.foot_ori = np.zeros(6)

    def get_foothold(self, pos, ori):
        self.foot_pos = pos
        self.foot_ori = ori

    def get_support_index(self, support_index):
        if support_index == 1:  # left support
            self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.u_ub = np.array([2.0 * self._m * self._g, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        if support_index == -1:  # right support
            self.u_lb = np.array([100., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.u_ub = np.array([2.0 * self._m * self._g, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def compute_cop_from_vertex(self, u):
        left_pos = self.foot_pos[:3]
        right_pos = self.foot_pos[3:]
        left_ori = self.foot_ori[:3]
        right_ori = self.foot_ori[3:]
        # needs to be modified
        cop = (np.tile(left_pos, (4, 1)).T + rotz(left_ori[2]).dot(roty(left_ori[1])).dot(rotx(left_ori[0])).dot(
            self.Vs.T)).dot(u[1:5]) + \
              (np.tile(right_pos, (3, 1)).T + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(
                  self.Vs[:3,:].T)).dot(u[5:]) + \
              (right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(self.Vs[3,:].T)).dot(1- sum(u[1:]))
        return cop

    def calc(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x  # how do you know cdot is the diff of c? implicit written in the source code of crocoddyl
        f_z = u.item(0)
        [u_x, u_y, u_z] = self.compute_cop_from_vertex(u)

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
        f_z = u.item(0)
        [u_x, u_y, u_z] = self.compute_cop_from_vertex(u)
        left_pos = self.foot_pos[:3]
        right_pos = self.foot_pos[3:]
        left_ori = self.foot_ori[:3]
        right_ori = self.foot_ori[3:]
        left_vertex_1 = left_pos + rotz(left_ori[2]).dot(roty(left_ori[1])).dot(rotx(left_ori[0])).dot(self.Vs[0, :])
        left_vertex_2 = left_pos + rotz(left_ori[2]).dot(roty(left_ori[1])).dot(rotx(left_ori[0])).dot(self.Vs[1, :])
        left_vertex_3 = left_pos + rotz(left_ori[2]).dot(roty(left_ori[1])).dot(rotx(left_ori[0])).dot(self.Vs[2, :])
        left_vertex_4 = left_pos + rotz(left_ori[2]).dot(roty(left_ori[1])).dot(rotx(left_ori[0])).dot(self.Vs[3, :])
        right_vertex_1 = right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(
            self.Vs[0, :])
        right_vertex_2 = right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(
            self.Vs[1, :])
        right_vertex_3 = right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(
            self.Vs[2, :])
        right_vertex_4 = right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(
            self.Vs[3, :])

        data.Fx[:, :] = np.array(
            [[
                f_z / ((c_z - u_z) * self._m), 0.0,
                -f_z * (c_x - u_x) / ((c_z - u_z) ** 2 * self._m), 0.0, 0.0, 0.0
            ],
                [
                    0.0, f_z / ((c_z - u_z) * self._m),
                         -f_z * (c_y - u_y) / ((c_z - u_z) ** 2 * self._m), 0.0, 0.0, 0.0
                ], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        df_dtau = np.array(
            [[(c_x - u_x) / (self._m * (c_z - u_z)),
              -f_z / ((c_z - u_z) * self._m), 0.,
              self._m * f_z * (c_x - u_x) / ((c_z - u_z) * self._m) ** 2],
             [(c_y - u_y) / (self._m * (c_z - u_z)), 0.,
              -f_z / ((c_z - u_z) * self._m),
              self._m * f_z * (c_y - u_y) / ((c_z - u_z) * self._m) ** 2],
             [1.0 / self._m, 0., 0., 0.]])  # needs to be modified
        # du_tmp = right_pos + rotz(right_ori[2]).dot(roty(right_ori[1])).dot(rotx(right_ori[0])).dot(right_vertex_4)
        dtau_du = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, left_vertex_1[0]-right_vertex_4[0], left_vertex_2[0]-right_vertex_4[0], left_vertex_3[0]-right_vertex_4[0], left_vertex_4[0]-right_vertex_4[0],
                             right_vertex_1[0]-right_vertex_4[0], right_vertex_2[0]-right_vertex_4[0], right_vertex_3[0]-right_vertex_4[0]],
                            [0.0, left_vertex_1[1]-right_vertex_4[1], left_vertex_2[1]-right_vertex_4[1], left_vertex_3[1]-right_vertex_4[1], left_vertex_4[1]-right_vertex_4[1],
                             right_vertex_1[1]-right_vertex_4[1], right_vertex_2[1]-right_vertex_4[1], right_vertex_3[1]-right_vertex_4[1]],
                            [0.0, left_vertex_1[2]-right_vertex_4[2], left_vertex_2[2]-right_vertex_4[2], left_vertex_3[2]-right_vertex_4[2], left_vertex_4[2]-right_vertex_4[2],
                             right_vertex_1[2]-right_vertex_4[2], right_vertex_2[2]-right_vertex_4[2], right_vertex_3[2]-right_vertex_4[2]]])  # 4*8
        data.Fu[:, :] = df_dtau.dot(dtau_du)

        data.contacts.contacts["single"].df_dx[:, :] = \
            np.array([[f_z / (c_z - u_z), 0.0, -f_z * (c_x - u_x) / ((c_z - u_z) ** 2), 0.0, 0.0, 0.0],
                      [0.0, f_z / (c_z - u_z), -f_z * (c_y - u_y) / ((c_z - u_z) ** 2), 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # needs to be modified
        # data.contacts.contacts["single"].df_du[:, :] = np.array([[(c_x - u_x) / (c_z - u_z), -f_z / (c_z - u_z), 0.0, f_z * (c_x - u_x) / ((c_z - u_z)**2)],
        #                                                          [(c_y - u_y) / (c_z - u_z), 0.0, -f_z / (c_z - u_z), f_z * (c_y - u_y) / ((c_z - u_z)**2)],

        contact_df_dtau = np.array(
            [[(c_x - u_x) / (c_z - u_z), -f_z / (c_z - u_z), 0.0, f_z * (c_x - u_x) / ((c_z - u_z) ** 2)],
             [(c_y - u_y) / (c_z - u_z), 0.0, -f_z / (c_z - u_z), f_z * (c_y - u_y) / ((c_z - u_z) ** 2)],
             [1.0, 0.0, 0.0, 0.0]])
        # contact_dtau_du =
        data.contacts.contacts["single"].df_du[:, :] = contact_df_dtau.dot(dtau_du)

        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        return DifferentialActionDataVariableHeightPendulum(self)


class DifferentialActionDataVariableHeightPendulum(
    crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = model.state.pinocchio.createData()
        self.contacts = model.contacts.createData(self.pinocchio)
        # self.pos =
        self.multibody = crocoddyl.DataCollectorMultibodyInContact(
            self.pinocchio, self.contacts)
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)

def buildSRBMFromRobot(robot_model):
    model = pinocchio.Model()
    placement = pinocchio.SE3.Identity()
    joint_com = model.addJoint(0, pinocchio.JointModelTranslation(), placement, 'com')  # for inverted pendulum model
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


# add constraints for sum of weight
def createPhaseModel(robot_model,
                     cop,
                     foot_pos,
                     foot_ori,
                     support=0,
                     xref=np.array([0.0, 0.0, 0.86, 0.1, 0.0, 0.0]),
                     nsurf=np.array([0., 0., 1.]).T,
                     mu=0.7,
                     Wx=np.array([0., 0., 10., 10., 10., 10.]),
                     Wu=np.array([50., 1., 1., 1., 1., 1., 1., 1.]),
                     wxreg=1,
                     wureg=1,
                     wutrack=50,
                     wxbox=1,
                     dt=4e-2):
    state = crocoddyl.StateVector(6)
    model = buildSRBMFromRobot(robot_model)
    multibody_state = crocoddyl.StateMultibody(model)
    runningCosts = crocoddyl.CostModelSum(state, 8)
    if support == 1: # Left support
        uRef = np.hstack(
            [9.81 * pinocchio.computeTotalMass(robot_model.model), 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0])
    elif support == 0: # Double support
        uRef = np.hstack(
            [9.81 * pinocchio.computeTotalMass(robot_model.model), 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    elif support == -1: # Right support
        uRef = np.hstack(
            [9.81 * pinocchio.computeTotalMass(robot_model.model), 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25])

    xRef = xref
    nSurf = nsurf
    Mu = mu
    ub_x = np.hstack([cop, np.zeros(3)]) + np.array(
        [0.3, 0.055, 0.95, 7., 7., 3])
    lb_x = np.hstack([cop, np.zeros(3)]) + np.array(
        [-0.3, -0.055, 0.75, -7., -7., -3])
    # --------------- COM Constraints ------------------ #
    runningCosts.addCost(
        "comBox",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(lb_x, ub_x)),
            crocoddyl.ResidualModelState(state, xRef, 8)), wxbox)
    # --------------- Track State Ref ------------------ #
    runningCosts.addCost(
        "comTrack",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelWeightedQuad(Wx),
            crocoddyl.ResidualModelState(state, xRef, 8)), wxreg)
    # --------------- Track Control Ref ------------------ #
    runningCosts.addCost("uTrack",
                         crocoddyl.CostModelResidual(
                             state, crocoddyl.ActivationModelWeightedQuad(Wu),
                             crocoddyl.ResidualModelControl(state, uRef)),
                         wutrack)  ## ||u||^2
    # --------------- Minimize Control ------------------ #
    runningCosts.addCost("uReg",
                         crocoddyl.CostModelResidual(
                             state, crocoddyl.ResidualModelControl(state, 8)),
                         wureg)  ## ||u||^2
    # --------------- Friction Cone Constraint ------------------ #
    cone = crocoddyl.FrictionCone(np.eye(3), Mu)
    runningCosts.addCost(
        "frictionPenalization",
        crocoddyl.CostModelResidual(
            multibody_state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)),
            crocoddyl.ResidualModelContactFrictionCone(multibody_state, 0,
                                                       cone, 8)), 1e2) #
    # --------------- Control Residual Constraint ------------------ #
    lb_dr = np.array([0.])
    ub_dr = np.array([1.])
    runningCosts.addCost(
        "Control Bound Constraint",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(lb_dr, ub_dr)),
            ControlResidual(state, 8)), 1e2)

    model = DifferentialActionModelVariableHeightPendulum(
        multibody_state, runningCosts)
    model.get_foothold(foot_pos, foot_ori)
    model.get_support_index(support)
    return crocoddyl.IntegratedActionModelEuler(model, dt)


def createTerminalModel(robot_model, cop, foot_pos, foot_ori):
    return createPhaseModel(robot_model,
                            cop,
                            foot_pos,
                            foot_ori,
                            xref=np.array([0.1, 0.0, 0.86, 0.0, 0.0, 0.0]),
                            Wx=np.array([50., 10., 50., 10., 10., 50.]),
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
    # u_x = [u[1] for u in us]
    # u_y = [u[2] for u in us]
    # u_z = [u[3] for u in us]

    plt.plot(cx)
    plt.show()
    plt.plot(cy)
    plt.show()
    plt.plot(cz)
    plt.show()
    # plt.plot(u_x)
    # plt.show()
    # plt.plot(u_y)
    # plt.show()


if __name__ == "__main__":

    foot_holds = np.array([[0.0, 0.0, 0.0], [0.0, -0.085, 0.0], [0.05, 0.0, 0.0],
                           [0.1, 0.085, 0.0], [0.1, 0.0, 0.0]]) # footsteps given
    phase = np.array([0, -1, 0, 1, 0])  # 0: double, 1: left, -1: right
    len_steps = phase.shape[0]
    robot_model = example_robot_data.load("talos")
    ################ Foot Placement #############################################
    foot_placements = np.zeros((len_steps, 6)) # :3, left support. 3:, right support
    foot_orientations = np.zeros((len_steps, 6))
    for i in range(len_steps):
        if phase[i] == 0:
            if i == 0 or i == len_steps-1:
                foot_placements[i, 0] = foot_holds[i, 0]
                foot_placements[i, 2] = foot_holds[i, 2]
                foot_placements[i, 3] = foot_holds[i, 0]
                foot_placements[i, 5] = foot_holds[i, 2]
                foot_placements[i, 1] = foot_holds[i, 1] + 0.085
                foot_placements[i, 4] = foot_holds[i, 1] - 0.085
            else:
                foot_placements[i, :] = foot_placements[i - 1, :]
        elif phase[i] == -1: # right support
            foot_placements[i, :3] = foot_holds[i+1, :]
            foot_placements[i, 3:] = foot_placements[i - 1, :3]
        elif phase[i] == 1: # left support
            foot_placements[i, 3:] = foot_holds[i+1, :]
            foot_placements[i, :3] = foot_placements[i - 1, 3:]
    print('Foot placements are ', foot_placements)
    num_nodes_single_support = 40
    num_nodes_double_support = 20
    dt = 4e-2
    support_indexes = phase
    support_durations = np.zeros(len_steps)
    for i in range(len_steps):
        if support_indexes[i] == 0:
            support_durations[i] = num_nodes_double_support * dt
        else:
            support_durations[i] = num_nodes_single_support * dt

    locoModel = [createPhaseModel(robot_model, foot_holds[0, :], foot_placements[0, :], foot_orientations[0, :])]
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
                                                 foot_placements[i, :],
                                                 foot_orientations[i, :],
                                                 support=phase[i],
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
                                                 foot_placements[i, :],
                                                 foot_orientations[i, :],
                                                 support=phase[i],
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
                                                 foot_placements[i, :],
                                                 foot_orientations[i, :],
                                                 support=phase[i],
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
                                              foot_placements[i, :],
                                              foot_orientations[i, :],
                                              support=phase[i],
                                              xref=np.array([
                                                  0.0, 0.0, 0.86 + tmp_cop[2], 0.1, 0.0, 0.0
                                              ]))
                         ] * num_nodes_single_support
            for j in range(num_nodes_single_support):
                cop = np.vstack((cop, foot_holds[i, :]))
    mT = createTerminalModel(robot_model, np.array([0.1, 0.0, 0.00]), foot_placements[-1, :], foot_orientations[-1, :])

    ##############################  Plot&Interpolate the data  #################################################
    # import matplotlib.pyplot as plt
    # plt.plot(cop[:,1])
    # plt.show()

    u_init = np.array([931.95, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    x_init = np.array([0.0, 0.0, 0.86, 0.0, 0.0, 0.0])
    problem = crocoddyl.ShootingProblem(x_init, locoModel, mT)
    solver = crocoddyl.SolverBoxFDDP(problem)
    log = crocoddyl.CallbackLogger()
    solver.setCallbacks([log, crocoddyl.CallbackVerbose()])
    t0 = time.time()
    solver.solve([x_init] * (problem.T + 1), [u_init] * problem.T,
                 5)  # x init, u init, max iteration
    print('Time of iteration consumed', time.time() - t0)
    plotComMotion(solver.xs, solver.us)

    # swing_height = 0.10
    # swing_height_offset = 0.01
    # com_gain = [30.0, 7.5, 1.0]
    #
    # com_pos = np.array([x[:3] for x in solver.xs])
    # com_vel = np.array([x[3:] for x in solver.xs])
    # com_acc = np.vstack((np.diff(com_vel, axis=0) / dt, np.zeros(3)))
    # total_time = np.cumsum(support_durations)[-1]
    # time = np.linspace(0, total_time, com_pos.shape[0])
    # # sampling based on the frequency
    # f_pos_x = interp1d(time,
    #                    com_pos[:, 0],
    #                    fill_value=(com_pos[0, 0], com_pos[-1, 0]),
    #                    bounds_error=False)
    # f_pos_y = interp1d(time,
    #                    com_pos[:, 1],
    #                    fill_value=(com_pos[0, 1], com_pos[-1, 1]),
    #                    bounds_error=False)
    # f_pos_z = interp1d(time,
    #                    com_pos[:, 2],
    #                    fill_value=(com_pos[0, 2], com_pos[-1, 2]),
    #                    bounds_error=False)
    # f_vel_x = interp1d(time,
    #                    com_vel[:, 0],
    #                    fill_value=(com_vel[0, 0], com_vel[-1, 0]),
    #                    bounds_error=False)
    # f_vel_y = interp1d(time,
    #                    com_vel[:, 1],
    #                    fill_value=(com_vel[0, 1], com_vel[-1, 1]),
    #                    bounds_error=False)
    # f_vel_z = interp1d(time,
    #                    com_vel[:, 2],
    #                    fill_value=(com_vel[0, 2], com_vel[-1, 2]),
    #                    bounds_error=False)
    # f_acc_x = interp1d(time,
    #                    com_acc[:, 0],
    #                    fill_value=(com_acc[0, 0], com_acc[-1, 0]),
    #                    bounds_error=False)
    # f_acc_y = interp1d(time,
    #                    com_acc[:, 1],
    #                    fill_value=(com_acc[0, 1], com_acc[-1, 1]),
    #                    bounds_error=False)
    # f_acc_z = interp1d(time,
    #                    com_acc[:, 2],
    #                    fill_value=(com_acc[0, 2], com_acc[-1, 2]),
    #                    bounds_error=False)
    # freq = 500
    # sample_num = int(freq * total_time)
    # time_sample = np.linspace(0, total_time, sample_num)
    #
    # com_traj_sample = np.vstack(
    #     (time_sample, f_pos_x(time_sample), f_pos_y(time_sample),
    #      f_pos_z(time_sample), f_vel_x(time_sample), f_vel_y(time_sample),
    #      f_vel_z(time_sample), f_acc_x(time_sample), f_acc_y(time_sample),
    #      f_acc_z(time_sample)))
    # # plt.plot(com_traj_sample[7, :])
    # # plt.show()
    #
    # import trajectory_publisher
    #
    # rospy.init_node('listener', anonymous=True)
    #
    # trajectory_publisher.publish_all(com_traj_sample.T, support_durations,
    #                                  support_indexes, foot_placements,
    #                                  foot_orientations, swing_height,
    #                                  swing_height_offset, com_gain)

    # np.savez('crocoddyl_data.npz', com_traj=com_traj_sample, support_durations=support_durations, support_indexes=support_indexes,
    #         foot_placements=foot_placements, foot_orientations=foot_orientations)
    # crocoddyl.plotOCSolution(log.xs[:], log.us)
    # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps)

    # np.save('../foot_holds.npy', np.concatenate((foot_holds, np.array([phase]).T), axis=1))
    # np.save('../com_traj.npy', solver.xs)
    # np.save('../control_traj.npy', solver.us)

    # solver.
