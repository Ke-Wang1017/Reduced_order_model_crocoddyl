crocoddyl_class/ActionModel.py : contains the simple lumped mass model

crocoddyl_class/ActionModel_nl.py : contains the non linear model
(levar arm in B matrix not constant, cf doc)

crocoddyl_class/GaitProblem.py : Create the ddp problem by using multiple nodes from 
ActionModel

To load non linear model in GaitProblem :

#Choose here whiche model : simple or non linear
from crocoddyl_class.ActionModel import *
#from crocoddyl_class.ActionModel_nl import * 

To launch the exemple : python3 exemple.py 
