from util import rotRollPitchYaw


class Vertex:
    def __init__(self, pos, ori, size, vertex_index):
        self.pos = pos
        self.ori = ori
        self.Vs = vertex_index * size / 2  # distance of 4 vertexes from the center of foot
        # self.p = np.zeros(3)

    def cal(self):
        return self.pos + rotRollPitchYaw(self.ori[0], self.ori[1], self.ori[2]).dot(self.Vs)


    # def createData(self, collector):
    #     return VertexData(self)
#
# class VertexData(crocoddyl.ResidualDataAbstract):
#     def __init__(self, collector):
#         crocoddyl.ResidualDataAbstract.__init__(self, collector)
#         self.p = np.zeros(3)