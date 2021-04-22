import crocoddyl
import numpy as np
import time
import matplotlib.pylab as plt
from crocoddyl_class.GaitProblem import *


####################
#  Initialization  #
####################

# Time step of the MPC (between nodes)
dt_mpc = 0.02

# Period of the MPC
T_mpc = 0.32

# Creation crocoddyl problem
gaitProblem = GaitProblem(mu = 0.7)
gaitProblem.createProblem()    


# Change the initial conditions here
# Initial velocity of Vx = 0.2m.s-1
lC = np.array([[0.0, 0.0, 0.2]]).T  # CoM centered and at 20 cm above the ground
abg = np.array([[0.0, 0.0, 0.0]]).T  # horizontal base (roll, pitch, 0.0)
lV = np.array([[0.2, 0.0, 0.0]]).T  # motionless base (linear velocity)
lW = np.array([[0.0, 0.0, 0.0]]).T  # motionless base (angular velocity)
l_feet = np.array([[0.19, 0.19, -0.19, -0.19],
                                 [0.15005, -0.15005, 0.15005, -0.15005],
                                 [0.0, 0.0, 0.0, 0.0]])  # position of feet in local frame


x0 = np.vstack((lC, abg,lV, lW))  # Current state vector
# The reference state, copy of the initial position,
xref = np.repeat(x0, np.int(T_mpc/dt_mpc)+1, axis=1)  # Desired future state vectors  
xref[6,:] = 0.0   # Target linear velocity to Vx = 0 


# Matrix representing the position of the feet 
# Gait matrix : Exemple :
# [[7 1 0 0 1],    7 period of dt_mpc in that position : FR, HL on the ground, others on swing phase
#  [1 1 1 1 1 ]]   1 period of dt_mpc of MPC in that position : all feet on the ground 
# fsteps : the same with position of the feet np.nan : swing phase
# [[7  px_1 py_1 pz1 nan nan nan ...   px4 py4 pz4],
#  []] 
fsteps = np.array([[ 1.00000000e+00,  1.90776486e-01,  1.48962816e-01,  4.22498932e-03,
   1.90060159e-01, -1.50265109e-01 , 0.00000000e+00 ,-1.89740429e-01,
   1.50467686e-01 , 1.78713224e-06 ,-1.90692335e-01 ,-1.48946056e-01,
   4.22561856e-03],
 [ 7.00000000e+00 , 1.90776486e-01 , 1.48962816e-01 , 4.22498932e-03,
              np.nan ,             np.nan   ,           np.nan   ,           np.nan,
               np.nan ,             np.nan ,-1.90692335e-01, -1.48946056e-01,
   4.22561856e-03],
 [ 1.00000000e+00,  1.90776486e-01,  1.48962816e-01,  4.22498932e-03,
   1.90000000e-01, -1.50050000e-01 , 0.00000000e+00 ,-1.90000000e-01,
   1.50050000e-01,  0.00000000e+00, -1.90692335e-01, -1.48946056e-01,
   4.22561856e-03],
 [ 7.00000000e+00  ,            np.nan  ,            np.nan       ,      np.nan,
   1.90000000e-01 ,-1.50050000e-01,  0.00000000e+00, -1.90000000e-01,
   1.50050000e-01,  0.00000000e+00 ,             np.nan      ,        np.nan,
               np.nan],
 [ 0.00000000e+00  ,            np.nan ,             np.nan,              np.nan,
               np.nan   ,           np.nan   ,  np.nan   ,           np.nan,
               np.nan    ,          np.nan ,             np.nan         ,     np.nan,
               np.nan],
 [ 0.00000000e+00   ,           np.nan     ,         np.nan      ,        np.nan,
               np.nan         ,     np.nan       ,       np.nan     ,         np.nan,
               np.nan        ,      np.nan,              np.nan ,             np.nan,
               np.nan]])


#############
#  Run MPC  #
#############


# update the mdynamic model for each nodes according to the ref state and the position of the feet
# that modify the B matrix (lever arm)

gaitProblem.updateProblem(fsteps , xref , x0)
gaitProblem.max_iter = 10

start_time = time.time()
gaitProblem.runProblem()

print("Temps d execution : %s secondes ---" % (time.time() - start_time))

# Rearrange the output of the crocoddyl solver
Xs = np.zeros((12,16))
Us = np.zeros((12,16)) 
for i in range (0,16) : 
    Xs[:,i] = gaitProblem.ddp.xs[i+1]
    Us[:,i] = gaitProblem.ddp.us[i]

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, T_mpc, np.int(T_mpc/dt_mpc))
l_str = ["X_osqp", "Y_osqp", "Z_osqp", "Roll_osqp", "Pitch_osqp", "Yaw_osqp", "Vx_osqp", "Vy_osqp", "Vz_osqp", "VRoll_osqp", "VPitch_osqp", "VYaw_osqp"]
l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    plt.plot(l_t, Xs[i,:], linewidth=2, marker='x' )
    plt.legend([l_str2[i]])

plt.suptitle("Predicted evolution of state variables")

# Desired evolution of contact forces
l_t = np.linspace(dt_mpc, T_mpc, np.int(T_mpc/dt_mpc))
l_str = ["FL_X_osqp", "FL_Y_osqp", "FL_Z_osqp", "FR_X_osqp", "FR_Y_osqp", "FR_Z_osqp", "HL_X_osqp", "HL_Y_osqp", "HL_Z_osqp", "HR_X_osqp", "HR_Y_osqp", "HR_Z_osqp"]
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    plt.plot(l_t, Us[i,:], linewidth=2, marker='x' )
    plt.legend([l_str2[i]])
plt.suptitle("Desired evolution of contact forces")

plt.show(block=True)
