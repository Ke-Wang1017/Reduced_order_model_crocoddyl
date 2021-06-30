import numpy as np
from util import rotx, roty, rotz, rotRollPitchYaw


class Vertex:
    def __init__(self, size, lpos, rpos, lori, rori):
        self.lfoot_pos = lpos
        self.rfoot_pos = rpos
        self.lfoot_ori = lori
        self.rfoot_ori = rori
        self.Vs = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1],
                            [1, -1, 1]]) * size / 2  # distance of 4 vertexes from the center of foot
        self.p = np.zeros(3)

    def cal_cop_from_vertex_humanoid(self, weight):
        # needs to be modified
        self.p[:] = (np.tile(self.lfoot_pos, (4, 1)).T + rotRollPitchYaw(self.lfoot_ori[0], self.lfoot_ori[1],
                    self.lfoot_ori[2]).dot(self.Vs.T)).dot(weight[1:5]) + \
                    (np.tile(self.rfoot_pos, (3, 1)).T + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1],
                    self.rfoot_ori[2]).dot(self.Vs[:3, :].T)).dot(weight[5:]) + \
                    (self.rfoot_pos + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1],
                                                      self.rfoot_ori[2]).dot(self.Vs[3, :].T)).dot(1 - sum(weight[1:]))
        return self.p

    def caldiff_cop(self):
        left_vertex_1 = self.lfoot_pos + rotRollPitchYaw(self.lfoot_ori[0], self.lfoot_ori[1], self.lfoot_ori[2]).dot(self.Vs[0, :])
        left_vertex_2 = self.lfoot_pos + rotRollPitchYaw(self.lfoot_ori[0], self.lfoot_ori[1], self.lfoot_ori[2]).dot(self.Vs[1, :])
        left_vertex_3 = self.lfoot_pos + rotRollPitchYaw(self.lfoot_ori[0], self.lfoot_ori[1], self.lfoot_ori[2]).dot(self.Vs[2, :])
        left_vertex_4 = self.lfoot_pos + rotRollPitchYaw(self.lfoot_ori[0], self.lfoot_ori[1], self.lfoot_ori[2]).dot(self.Vs[3, :])
        right_vertex_1 = self.rfoot_pos + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1], self.rfoot_ori[2]).dot(self.Vs[0, :])
        right_vertex_2 = self.rfoot_pos + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1], self.rfoot_ori[2]).dot(self.Vs[1, :])
        right_vertex_3 = self.rfoot_pos + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1], self.rfoot_ori[2]).dot(self.Vs[2, :])
        right_vertex_4 = self.rfoot_pos + rotRollPitchYaw(self.rfoot_ori[0], self.rfoot_ori[1], self.rfoot_ori[2]).dot(self.Vs[3, :])

        dtau_du = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, left_vertex_1[0]-right_vertex_4[0], left_vertex_2[0]-right_vertex_4[0], left_vertex_3[0]-right_vertex_4[0], left_vertex_4[0]-right_vertex_4[0],
                             right_vertex_1[0]-right_vertex_4[0], right_vertex_2[0]-right_vertex_4[0], right_vertex_3[0]-right_vertex_4[0]],
                            [0.0, left_vertex_1[1]-right_vertex_4[1], left_vertex_2[1]-right_vertex_4[1], left_vertex_3[1]-right_vertex_4[1], left_vertex_4[1]-right_vertex_4[1],
                             right_vertex_1[1]-right_vertex_4[1], right_vertex_2[1]-right_vertex_4[1], right_vertex_3[1]-right_vertex_4[1]],
                            [0.0, left_vertex_1[2]-right_vertex_4[2], left_vertex_2[2]-right_vertex_4[2], left_vertex_3[2]-right_vertex_4[2], left_vertex_4[2]-right_vertex_4[2],
                             right_vertex_1[2]-right_vertex_4[2], right_vertex_2[2]-right_vertex_4[2], right_vertex_3[2]-right_vertex_4[2]]])  # 4*8

        return dtau_du

    # def cal_cop_from_vertex_quadruped(self):
