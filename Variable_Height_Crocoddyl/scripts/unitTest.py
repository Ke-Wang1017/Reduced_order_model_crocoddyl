import crocoddyl
import numpy as np
from variable_height_analytical import *

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
        raise NumDiffException("NumDiff exception, with residual of %.4g, above threshold %.4g" % (value, threshold))



model = DifferentialActionModelVariableHeightPendulum(costs)
data = model.createData()

mnum = crocoddyl.DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()

xrand = np.random.rand(6)
urand = np.random.rand(1)
model.calc(data, xrand, urand)
model.calcDiff(data, xrand, urand)
mnum.calc(dnum, xrand, urand)
mnum.calcDiff(dnum, xrand, urand)

NUMDIFF_MODIFIER = 3e4
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu, NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu, NUMDIFF_MODIFIER * mnum.disturbance)
