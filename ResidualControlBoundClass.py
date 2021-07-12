import crocoddyl


class ControlBoundResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu) # , False, False, True

    def calc(self, data, x, u):
        data.r[:] = 1 - sum(u[1:])

    def calcDiff(self, data, x, u):
        pass

    def createData(self, collector):
        data = crocoddyl.ResidualDataAbstract(self, collector)
        data.Ru[1:] = -1
        return data
