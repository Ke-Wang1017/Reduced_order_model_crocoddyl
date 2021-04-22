# coding: utf8

import crocoddyl
import numpy as np
import utils


class ActionModel(crocoddyl.ActionModelAbstract):

    def __init__(self,nx = 12 ,nu = 12 , mu = 0.8 , log = True ):

        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu, 1)      
        #:param state: state description,
        #:param nu: dimension of control vector,
        #:param nr: dimension of the cost-residual vector (default 1)

        # Time step of the solver
        self.dt = 0.02

        # Mass of the robot
        self.mass = 2.97784899

        # Inertia matrix of the robot in body frame (found in urdf)
        self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])

        # Friction coefficient
        self.mu = mu

        # Matrix A -- X_k+1 = AX + BU --
        self.A = np.zeros((12, 12))

        # Matrix B
        self.B = np.zeros((12, 12))

        # Vector xref
        self.xref = np.zeros(12)
        
        # Vector g
        self.g = np.zeros((12,))

        # Weight vector for the state
        self.stateWeight = np.full(12,2)

        # Weight on the friction cost
        self.frictionWeight = 0.1
        
        # Weight on the command forces
        self.weightForces = np.full(12,0.1)

        # Levers Arms used in B
        self.lever_arms = np.zeros((3,4))
        

        # Initial position of footholds in the "straight standing" default configuration
        # footholds = [[ px_foot1 , px_foot2 ...     ] ,
        #              [ py_foot1 , py_foot2 ...     ] ,
        #              [ pz_foot1 , pz_foot2 ...     ] ]  2D -- pz = 0
        self.footholds = np.array(
            [[0.19, 0.19, -0.19, -0.19],
             [0.15005, -0.15005, 0.15005, -0.15005],
             [0.0, 0.0, 0.0, 0.0]])

        # S matrix 
        # Represents the feet that are in contact with the ground such as : 
        # S = [1 0 0 1] --> FL(Front Left) in contact, FR not , HL not , HR in contact
        self.S = np.ones(4)

        # List of the feet in contact with the ground used to compute B
        # if S = [1 0 0 1] ; L_feet = [1 1 1 0 0 0 0 0 0 1 1 1]
        self.L_feet = np.zeros(12)

        #Normal vector for friction cone
        self.nsurf = np.matrix([0., 0., 1.]).T

        # Cone croccodyl class
        self.cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)

        self.lb = np.tile(self.cone.lb , 4)
        self.ub = np.tile(self.cone.ub , 4)

        # Matrice Fa of the cone class, repeated 4 times (4 feet)
        self.Fa =  np.zeros((20,12))

        # Inequality activation model.
        #  The activation is zero when r is between the lower (lb) and upper (ub) bounds, beta
        #  determines how much of the total range is not activated. This is the activation
        #  equations:
        #  a(r) = 0.5 * ||r||^2 for lb < r < ub
        #  a(r) = 0. for lb >= r >= ub.        
        self.activation = crocoddyl.ActivationModelQuadraticBarrier((crocoddyl.ActivationBounds(self.lb, self.ub)))
        
        self.dataCost = self.activation.createData()
        
        self.Lu_f = np.zeros(12) 
        
        self.Luu_f = np.zeros((12,12))

        # Log parameters
        self.log = log 
        self.log_cost_friction = []
        self.log_cost_state = []
        self.log_state = []
        self.log_u = []
        
        # Other dynamic models
        self.I_inv = np.zeros((3,3))
        self.derivative_B = np.zeros((12,12))  # Here lever arm is constant, not used
        

    def calc(self, data, x, u):
        #It describes the time-discrete evolution of our dynamical system
        #in which we obtain the next state. Additionally it computes the
        #cost value associated to this discrete state and control pair.
        #:param data: action data
        #:param x: time-discrete state vector
        #:param u: time-discrete control input

        data.xnext[:] = np.dot(self.A,x) + np.dot(self.B,u) + self.g

        # Compute friction cone 
        self.costFriction(u)

        data.r = np.concatenate([self.stateWeight*(x-self.xref) , self.weightForces*u ])

        data.cost = .5* (sum((data.r)**2).item()) +  self.frictionWeight*self.dataCost.a_value

        if self.log : 
            self.log_cost_friction.append(self.frictionWeight*self.dataCost.a_value)
            self.log_cost_state.append(.5* (sum((data.r)**2).item()))
            self.log_state.append(u)
            self.log_u.append(u)


    def calcDiff(self,data,x,u):
        #It computes the partial derivatives of the dynamical system and the
        #cost function. It assumes that calc has been run first.
        #This function builds a quadratic approximation of the
        #action model (i.e. linear dynamics and quadratic cost).
        #:param data: action data
        #:param x: time-discrete state vector
        #:param u: time-discrete control input


        data.Fx[:,:] = self.A[:,:] + self.derivative_B[:,:]   # here derivative B is null
        data.Fu[:,:] = self.B[:,:]
        data.Lxx[:,:] = np.diag(self.stateWeight**2)[:,:]
        data.Lx[:] = self.stateWeight*data.r[0:12]
        data.Lu[:] =  self.frictionWeight*self.Lu_f + self.weightForces*data.r[12:]
        data.Luu[:,:] =  self.frictionWeight*self.Luu_f + np.diag(self.weightForces**2)[:,:]

    def createModel(self):
        ''' Creation of the dynamic model
        '''


        # Create matrix A
        self.A = np.eye(12)
        self.A[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] = np.ones((6,)) * self.dt

         # Create matrix B
        self.B = np.zeros((12, 12))

        # Create matrix g
        self.g[8] = -9.81 * self.dt

        # Create Fa matrix
        for k in range(4) : 
            self.Fa[5*k:5*k+5 , 3*k:3*k+3] = self.cone.A

    def updateModel(self,l_feet,xref,S = np.ones(4)):
        ''' Update the dynamic model 
        Args :
        l_feet (x Array 3x4) : position of the feet 
        xref  (array x12) : Ref trajectory (pos, orientation ,lin vel, angular vel)
        S    (array x4) : S represents the sequence of feet touching the ground
                          S = [1 0 0 1] --> FL in contact, FR not , HL not , HR in contact
        '''

        self.xref = xref
        self.S = S

        # Footholds position in the local frame
        self.footholds[0:2, :] = l_feet[0:2, :]

        # No interaction in the dynamic model for the feets  that are not in contact with the ground
        # Use of the matrix S which represents feet on the ground
        
        for i in range(len(self.S)) : 
            self.L_feet[[3*i,3*i+1,3*i+2]] = self.S[i] 
        
        self.B[np.tile([6, 7, 8], 4), np.arange(0, 12, 1)] = (self.dt / self.mass) * self.L_feet

        # Get inverse of the inertia matrix 
        c, s = np.cos(self.xref[5]), np.sin(self.xref[5])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.I_inv = np.linalg.inv(np.dot(R, self.gI))

        # Get skew-symetric matrix for each foothold
        self.lever_arms = self.footholds - np.matrix(self.xref[0:3]).transpose()
        for i in range(4): 
            if(self.S[i] != 0 ) : 
                self.B[-3:, (i*3):((i+1)*3)] = self.dt * np.dot(self.I_inv, utils.getSkew(self.lever_arms[:, i]))
            else : 
                # Feet not in contact with the ground 
                self.B[-3:, (i*3):((i+1)*3)] = np.zeros((3,3))

        self.log_cost_friction = []
        self.log_cost_state = []
        self.log_state = []
        self.log_u = []
    

    def costFriction(self,u) : 

        self.activation.calc(self.dataCost , np.dot(self.Fa,u))
        self.activation.calcDiff(self.dataCost , np.dot(self.Fa,u))
        self.Lu_f = self.Fa.transpose() @ self.dataCost.Ar
        self.Luu_f = np.dot(self.Fa.transpose() , self.dataCost.Arr) @ self.Fa



    

    


            

            
      






        

        

