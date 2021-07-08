import numpy as np
import crocoddyl


class ActuationModelBipedContactVertex(crocoddyl.ActuationModelAbstract):
    def __init__(self):
        crocoddyl.ActuationModelAbstract.__init__(4)
        self._vs = np.zeros((8, 3)) # bipedal robot, 8 vertexes and each has 3D coordinate
        self._updateVertexes = True

    def set_reference(self, vertexes):
        if vertexes.shape[0] == 8 and vertexes.shape[1] == 3:
            self._vs = vertexes
            self._updateVertexes = True
        else:
            raise Exception('Wrong shape of vertexes!')

    def calc(self, data, x, u): # tau: f_z, cop_x, cop_y, cop_z
        data.tau[0] = u[0]
        w = u[1:]
        data.tau[1:] = (self._vs[:4, :]).dot(w[1:5])
        data.tau[1:] += (self._vs[4:7, :]).dot(w[5:])
        data.tau[1:] += (self._vs[-1, :]).dot(1 - sum(w[1:]))

    def calcDiff(self, data, x, u):
        self.updateJacobians(data)
        pass

    def updateJacobians(self, data):
        if self._updateVertexes:
            self._updateVertexes = False
            data.dtau_du[0, 0] = 1  # note that the other columns are zero by default
            data.dtau_du[1, 1:] = [self.set[0, 0] - self.set[-1, 0], self.set[1, 0] - self.set[-1, 0],
                                   self.set[2, 0] - self.set[-1, 0], self.set[3, 0] - self.set[-1, 0],
                                   self.set[4, 0] - self.set[-1, 0], self.set[5, 0] - self.set[-1, 0],
                                   self.set[6, 0] - self.set[-1, 0]]
            data.dtau_du[2, 1:] = [self.set[0, 1] - self.set[-1, 1], self.set[1, 1] - self.set[-1, 1],
                                  self.set[2, 1] - self.set[-1, 1], self.set[3, 1] - self.set[-1, 1],
                                  self.set[4, 1] - self.set[-1, 1], self.set[5, 1] - self.set[-1, 1],
                                  self.set[6, 1] - self.set[-1, 1]]
            data.dtau_du[3, 1:] = [self.set[0, 2] - self.set[-1, 2], self.set[1, 2] - self.set[-1, 2],
                                  self.set[2, 2] - self.set[-1, 2], self.set[3, 2] - self.set[-1, 2],
                                  self.set[4, 2] - self.set[-1, 2], self.set[5, 2] - self.set[-1, 2],
                                  self.set[6, 2] - self.set[-1, 2]]

      def createData(self):
        data = crocoddyl.ActuationDataAbstract(self)
        self.updateJacobians(data)
        return data