import crocoddyl
import numpy as np
from util import rotx, roty, rotz

class AsymmetricFrictionConeResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu)
        self.friction_x_p = np.zeros(3)
        self.friction_x_n = np.zeros(3)
        self.friction_y_p = np.zeros(3)
        self.friction_y_n = np.zeros(3)
        foot_size = [0.2, 0.1, 0]
        self.Vs = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1],
                            [1, -1, 1]]) * foot_size / 2
        self.foot_pos = np.zeros(6)
        self.foot_ori = np.zeros(6)

    def getCone(self, x_p, x_n, y_p, y_n):
        self.friction_x_p = x_p
        self.friction_x_n = x_n
        self.friction_y_p = y_p
        self.friction_y_n = y_n

    def get_foothold(self, pos, ori):
        self.foot_pos = pos
        self.foot_ori = ori

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
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        [u_x, u_y, u_z] = self.compute_cop_from_vertex(u)
        force = np.array([(c_x-u_x)/(c_z-u_z), (c_y-u_y)/(c_z-u_z), 1.0])
        data.r[:] = -np.array([self.friction_x_p.dot(force), self.friction_x_n.dot(force), self.friction_y_p.dot(force),\
                              self.friction_y_n.dot(force)])

    def calcDiff(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
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
        dtau_du = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, left_vertex_1[0]-right_vertex_4[0], left_vertex_2[0]-right_vertex_4[0], left_vertex_3[0]-right_vertex_4[0], left_vertex_4[0]-right_vertex_4[0],
                             right_vertex_1[0]-right_vertex_4[0], right_vertex_2[0]-right_vertex_4[0], right_vertex_3[0]-right_vertex_4[0]],
                            [0.0, left_vertex_1[1]-right_vertex_4[1], left_vertex_2[1]-right_vertex_4[1], left_vertex_3[1]-right_vertex_4[1], left_vertex_4[1]-right_vertex_4[1],
                             right_vertex_1[1]-right_vertex_4[1], right_vertex_2[1]-right_vertex_4[1], right_vertex_3[1]-right_vertex_4[1]],
                            [0.0, left_vertex_1[2]-right_vertex_4[2], left_vertex_2[2]-right_vertex_4[2], left_vertex_3[2]-right_vertex_4[2], left_vertex_4[2]-right_vertex_4[2],
                             right_vertex_1[2]-right_vertex_4[2], right_vertex_2[2]-right_vertex_4[2], right_vertex_3[2]-right_vertex_4[2]]])  # 4*8
        # Rx is 4x6
        data.Rx[:,:] = \
            -np.array([[self.friction_x_p[0]/(c_z - u_z), self.friction_x_p[1]/(c_z - u_z), -self.friction_x_p[0]*(c_x-u_x)/((c_z-u_z)**2)-self.friction_x_p[1]*(c_y-u_y)/((c_z-u_z)**2), 0.0, 0.0, 0.0],
                       [self.friction_x_n[0]/(c_z - u_z), self.friction_x_n[1]/(c_z - u_z), -self.friction_x_n[0]*(c_x-u_x)/((c_z-u_z)**2)-self.friction_x_n[1]*(c_y-u_y)/((c_z-u_z)**2), 0.0, 0.0, 0.0],
                       [self.friction_y_p[0]/(c_z - u_z), self.friction_y_p[1]/(c_z - u_z), -self.friction_y_p[0]*(c_x-u_x)/((c_z-u_z)**2)-self.friction_y_p[1]*(c_y-u_y)/((c_z-u_z)**2), 0.0, 0.0, 0.0],
                       [self.friction_y_n[0]/(c_z - u_z), self.friction_y_n[1]/(c_z - u_z), -self.friction_y_n[0]*(c_x-u_x)/((c_z-u_z)**2)-self.friction_y_n[1]*(c_y-u_y)/((c_z-u_z)**2), 0.0, 0.0, 0.0]])
        # Ru is 4x8
        dr_dtau = \
            -np.array([[0, -self.friction_x_p[0]/(c_z-u_z), -self.friction_x_p[1]/(c_z-u_z), self.friction_x_p[0]*(c_x-u_x)/((c_z-u_z)**2)+self.friction_x_p[1]*(c_y-u_y)/((c_z-u_z)**2)],
                       [0, -self.friction_x_n[0]/(c_z-u_z), -self.friction_x_n[1]/(c_z-u_z), self.friction_x_n[0]*(c_x-u_x)/((c_z-u_z)**2)+self.friction_x_n[1]*(c_y-u_y)/((c_z-u_z)**2)],
                       [0, -self.friction_y_p[0]/(c_z-u_z), -self.friction_y_p[1]/(c_z-u_z), self.friction_y_p[0]*(c_x-u_x)/((c_z-u_z)**2)+self.friction_y_p[1]*(c_y-u_y)/((c_z-u_z)**2)],
                       [0, -self.friction_y_n[0]/(c_z-u_z), -self.friction_y_n[1]/(c_z-u_z), self.friction_y_n[0]*(c_x-u_x)/((c_z-u_z)**2)+self.friction_y_n[1]*(c_y-u_y)/((c_z-u_z)**2)]])
        data.Ru = dr_dtau.dot(dtau_du)

    def createData(self, collector):
        data = crocoddyl.ResidualDataAbstract(self, collector)
        return data

