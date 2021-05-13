import crocoddyl
import numpy as np
from variable_height_analytical import DifferentialActionModelVariableHeightPendulum, DifferentialActionDataVariableHeightPendulum


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


state = crocoddyl.StateVector(6)
weights = np.array([0., 0., 10., 0., 50., 0.])
xRef = np.array([0.0, 0.0, 0.98, 0.0, 0.0, 0.0])
runningCosts = crocoddyl.CostModelSum(state, 4)
runningCosts.addCost(
    "comTracking",
    crocoddyl.CostModelState(state,
                             crocoddyl.ActivationModelWeightedQuad(weights),
                             xRef, 4), 1e3)
runningCosts.addCost("uReg", crocoddyl.CostModelControl(state, 4),
                     1e-3)  ## ||u||^2
model = DifferentialActionModelVariableHeightPendulum(runningCosts)
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
