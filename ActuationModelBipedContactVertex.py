import numpy as np
import crocoddyl
import copy


class ActuationModelBipedContactVertex(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 8)
        self._vs = np.zeros((3, 8))  # bipedal robot, 8 vertexes and each has 3D coordinate
        self._updateVertexes = True

    def calc(self, data, x, u):  # tau: cop_x, cop_y, cop_z
        data.tau[:] = (self._vs[:, :7]).dot(u[1:])
        data.tau[:] += (self._vs[:, -1]).dot(1 - sum(u[1:]))

    def calcDiff(self, data, x, u):
        self.updateJacobians(data)
        pass

    def createData(self):
        data = crocoddyl.ActuationDataAbstract(self)
        self._updateVertexes = True
        self.updateJacobians(data)
        return data

    def updateJacobians(self, data):
        if self._updateVertexes:
            self._updateVertexes = False
            data.dtau_du[0, 1:] = [
                self._vs[0, 0] - self._vs[0, -1], self._vs[0, 1] - self._vs[0, -1], self._vs[0, 2] - self._vs[0, -1],
                self._vs[0, 3] - self._vs[0, -1], self._vs[0, 4] - self._vs[0, -1], self._vs[0, 5] - self._vs[0, -1],
                self._vs[0, 6] - self._vs[0, -1]
            ]
            data.dtau_du[1, 1:] = [
                self._vs[1, 0] - self._vs[1, -1], self._vs[1, 1] - self._vs[1, -1], self._vs[1, 2] - self._vs[1, -1],
                self._vs[1, 3] - self._vs[1, -1], self._vs[1, 4] - self._vs[1, -1], self._vs[1, 5] - self._vs[1, -1],
                self._vs[1, 6] - self._vs[1, -1]
            ]
            data.dtau_du[2, 1:] = [
                self._vs[2, 0] - self._vs[2, -1], self._vs[2, 1] - self._vs[2, -1], self._vs[2, 2] - self._vs[2, -1],
                self._vs[2, 3] - self._vs[2, -1], self._vs[2, 4] - self._vs[2, -1], self._vs[2, 5] - self._vs[2, -1],
                self._vs[2, 6] - self._vs[2, -1]
            ]

    def set_reference(self, vertexes):
        if vertexes.shape[0] == 3 and vertexes.shape[1] == 8:
            self._vs = copy.deepcopy(vertexes)
            self._updateVertexes = True
        else:
            raise Exception('Wrong shape of vertexes!')
