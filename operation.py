from sys import path
path.append(r"/home/zeiros/MPC_general_model/Macenum_Model")
from Macenum_Model import MACE

mace = MACE(0.1,0.2,0.1,0,0,0,15,15,0,0.2,20)
mace.set_weights(9,7,1,0.05,0.05,0.05,0.05)
mace.init_variables()
mace.operate()

