import crocoddyl
import numpy as np
from variable_height_analytical_3D_CoP import DifferentialActionModelVariableHeightPendulum, buildSRBMFromRobot
import example_robot_data
import pinocchio

class NumDiffException(Exception):
    """Raised when the NumDiff values are too high"""
    pass


def assertNumDiff(A, B, threshold):
    """ Assert analytical derivatives against NumDiff using the error norm.
    :param A: analytical derivatives
    :param B: NumDiff derivatives
    :param threshold: absolute tolerance
    """
    if not np.allclose(A, B, atol=threshold):
        value = np.linalg.norm(A - B)
        raise NumDiffException(
            "NumDiff exception, with residual of %.4g, above threshold %.4g" %
            (value, threshold))


cop = np.zeros(3)
xRef = np.array([0.0, 0.0, 0.86, 0.15, 0.0, 0.0])
nSurf = np.array([0., 0., 1.]).T
Wx = np.array([0., 0., 10., 10., 10., 10.])
Wu = np.array([0., 50., 50., 1.])
wxreg = 1
wureg = 5
wutrack = 50
wxbox = 1
dt = 2e-2
state = crocoddyl.StateVector(6)
robot = example_robot_data.load("talos")
model = buildSRBMFromRobot(robot)
multibody_state = crocoddyl.StateMultibody(model)
runningCosts = crocoddyl.CostModelSum(state, 4)
uRef = np.hstack([np.zeros(1), cop])
Mu = 0.7
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
model = DifferentialActionModelVariableHeightPendulum(multibody_state, runningCosts)
data = model.createData()

mnum = crocoddyl.DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()

xrand = np.random.rand(6)
urand = np.random.rand(4)
model.calc(data, xrand, urand)
model.calcDiff(data, xrand, urand)
mnum.calc(dnum, xrand, urand)
mnum.calcDiff(dnum, xrand, urand)

NUMDIFF_MODIFIER = 3e4
assertNumDiff(
    data.Fx, dnum.Fx, NUMDIFF_MODIFIER * mnum.disturbance
)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu, NUMDIFF_MODIFIER *
              mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
assertNumDiff(
    data.Lx, dnum.Lx, NUMDIFF_MODIFIER * mnum.disturbance
)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu, NUMDIFF_MODIFIER * mnum.disturbance)
