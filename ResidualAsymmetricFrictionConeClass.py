import crocoddyl
import numpy as np
from math import sqrt
from util import rotRollPitchYaw
from ContactVertexClass_backup import Vertex

class AsymmetricFrictionConeResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, ori, mu, vertex, vertex_data):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu)
        self.friction_x_p = np.zeros(3)
        self.friction_x_n = np.zeros(3)
        self.friction_y_p = np.zeros(3)
        self.friction_y_n = np.zeros(3)
        self.vertex = vertex
        self.vertex_data = vertex_data
        self.ori = ori
        self.mu = mu
        self.p = np.zeros(3)
        self.compute_friction_cones()

    def compute_friction_cones(self):
        x_p = np.array([self.mu / (sqrt(self.mu ** 2 + 1)), 0, 1 / (sqrt(self.mu ** 2 + 1))])
        x_n = np.array([-self.mu / (sqrt(self.mu ** 2 + 1)), 0, 1 / (sqrt(self.mu ** 2 + 1))])
        y_p = np.array([0, self.mu / (sqrt(self.mu ** 2 + 1)), 1 / (sqrt(self.mu ** 2 + 1))])
        y_n = np.array([0, -self.mu / (sqrt(self.mu ** 2 + 1)), 1 / (sqrt(self.mu ** 2 + 1))])
        R_L = rotRollPitchYaw(self.ori[0], self.ori[1], self.ori[2])
        R_R = rotRollPitchYaw(self.ori[3], self.ori[4], self.ori[5])
        x_p_l = R_L.dot(x_p)
        x_n_l = R_L.dot(x_n)
        y_p_l = R_L.dot(y_p)
        y_n_l = R_L.dot(y_n)

        x_p_r = R_R.dot(x_p)
        x_n_r = R_R.dot(x_n)
        y_p_r = R_R.dot(y_p)
        y_n_r = R_R.dot(y_n)

        if x_p_r[2] > x_p_l[2]:
            self.friction_x_p = x_p_l
        else:
            self.friction_x_p = x_p_r
        if x_n_r[2] > x_n_l[2]:
            self.friction_x_n = x_n_l
        else:
            self.friction_x_n = x_n_r
        if y_p_r[2] > y_p_l[2]:
            self.friction_y_p = y_p_l
        else:
            self.friction_y_p = y_p_r
        if y_n_r[2] > y_n_l[2]:
            self.friction_y_n = y_n_l
        else:
            self.friction_y_n = y_n_r

    def calc(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        self.p = self.vertex.cal(self.vertex_data, u)
        [u_x, u_y, u_z] = self.p
        force = np.array([(c_x-u_x)/(c_z-u_z), (c_y-u_y)/(c_z-u_z), 1.0])
        data.r[:] = -np.array([self.friction_x_p.dot(force), self.friction_x_n.dot(force), self.friction_y_p.dot(force),\
                              self.friction_y_n.dot(force)])

    def calcDiff(self, data, x, u):
        c_x, c_y, c_z, cdot_x, cdot_y, cdot_z = x
        [u_x, u_y, u_z] = self.p
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

        data.Ru = dr_dtau.dot(self.vertex_data.dtau_du)

    def createData(self, collector):
        data = crocoddyl.ResidualDataAbstract(self, collector)
        return data

