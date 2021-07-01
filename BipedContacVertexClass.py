import numpy as np

import crocoddyl
from ContactVertexClass import Vertex, VertexData

class BipedContactVertex:
    def __init__(self):
        self.vertex_set = np.zeros((8,3))
        self.cop = np.zeros(3)

    def get_vertex(self, vertex_set):
        self.vertex_set = vertex_set

    def calc(self, weight):
        self.cop[:] = (self.vertex_set[:4,:]).dot(weight[1:5]) + \
                    (self.vertex_set[4:7,:]).dot(weight[5:]) + \
                    (self.vertex_set[-1,:]).dot(1 - sum(weight[1:]))
        return self.cop

    def calcdiff(self):
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, self.vertex_set[0, 0]-self.vertex_set[-1, 0], self.vertex_set[1, 0]-self.vertex_set[-1, 0],
                             self.vertex_set[2, 0]-self.vertex_set[-1, 0], self.vertex_set[3, 0]-self.vertex_set[-1, 0],
                             self.vertex_set[4, 0]-self.vertex_set[-1, 0], self.vertex_set[5, 0]-self.vertex_set[-1, 0],
                              self.vertex_set[6, 0]-self.vertex_set[-1, 0]],
                            [0.0, self.vertex_set[0, 1]-self.vertex_set[-1, 1], self.vertex_set[1, 1]-self.vertex_set[-1, 1],
                             self.vertex_set[2, 1]-self.vertex_set[-1, 1], self.vertex_set[3, 1]-self.vertex_set[-1, 1],
                             self.vertex_set[4, 1]-self.vertex_set[-1, 1], self.vertex_set[5, 1]-self.vertex_set[-1, 1],
                             self.vertex_set[6, 1]-self.vertex_set[-1, 1]],
                            [0.0, self.vertex_set[0, 2]-self.vertex_set[-1, 2], self.vertex_set[1, 2]-self.vertex_set[-1, 2],
                             self.vertex_set[2, 2]-self.vertex_set[-1, 2], self.vertex_set[3, 2]-self.vertex_set[-1, 2],
                             self.vertex_set[4, 2]-self.vertex_set[-1, 2], self.vertex_set[5, 2]-self.vertex_set[-1, 2],
                             self.vertex_set[6, 2]-self.vertex_set[-1, 2]]])  # 4*8