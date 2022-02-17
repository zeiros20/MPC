from sys import path
path.append(r"/home/zeiros/mpc_double_model/Sim_code")
path.append(r"/home/zeiros/casadi-linux-py38-v3.5.5-64bit")
path.append(r"/home/zeiros/MPC_general_Model/General")
from General import MPC
import casadi as ca
from casadi import *
import math
import numpy as np
import matplotlib.pyplot as plt
from time import time

class MACE(MPC):
    def __init__(self,Rw,L,l,x_init, y_init, theta_init, x_target, y_target, theta_target,h,N,omega_max = math.pi):
        super().__init__(x_init, y_init, theta_init, x_target, y_target, theta_target,h,N)
        self.__Rw = Rw
        self.__L = L
        self.__l = l
        self.__omega_max = omega_max
        self.__omega_min = -omega_max


    def set_weights(self, Q_X, Q_Y, Q_theta, R_w1, R_w2, R_w3, R_w4):
        self.__Q = ca.diagcat(Q_X, Q_Y, Q_theta)
        self.__R = ca.diagcat(R_w1, R_w2, R_w3, R_w4)
        

    

    
    def init_variables(self):
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        self.__theta = ca.SX.sym('theta')
        self.__states = ca.vertcat(x,y,self.__theta)

        omega_W1 = ca.SX.sym('omega_W1')
        omega_W2 = ca.SX.sym('omega_W2')
        omega_W3 = ca.SX.sym('omega_W3')
        omega_W4 = ca.SX.sym('omega_W4')
        self.__controls = ca.vertcat(omega_W1, omega_W2, omega_W3, omega_W4)
    

    def set_rhs(self):
        rot_3d_z = ca.vertcat(
        ca.horzcat(np.cos(self.__theta), -np.sin(self.__theta), 0),
        ca.horzcat(np.sin(self.__theta), np.cos(self.__theta), 0),
        ca.horzcat(0, 0, 1)
            )
        j0_plus = (self.__Rw / 4) * ca.DM([
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1 / (self.__L + self.__l), 1 / (self.__L + self.__l), -1 / (self.__L + self.__l), 1 / (self.__L + self.__l)]
            ])
        rhs = rot_3d_z @ j0_plus @ self.__controls
        return rhs

    def obj_fin(self):
        rhs = self.set_rhs()
        pack = self.generate_expressions(self.__states, self.__controls, rhs)
        f, X, U ,P = pack
        self.__pack = pack
        obj = 0
        n_states = self.__states.numel()
        g = X[:, 0] - P[:n_states]
        for k in range(self.__N):
            st = X[:, k]
            con = U[:, k]
            obj = obj + (st - P[n_states:]).T @ self.__Q @ (st - P[n_states:]) + con.T @ self.__R @ con
            st_next = X[:, k + 1]
            k1 = f(st, con)
            k2 = f(st + self.__h / 2 * k1, con)
            k3 = f(st + self.__h / 2 * k2, con)
            k4 = f(st + self.__h * k3, con)
            st_next_RK4 = st + (self.__h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)
        self.__obj = obj
        self.__g = g 

        lbx = ca.DM.zeros((n_states * (self.__N + 1) + self.__controls.numel() * self.__N, 1))
        ubx = ca.DM.zeros((n_states * (self.__N + 1) + self.__controls.numel() * self.__N, 1))

        lbx[0: n_states * (self.__N + 1): n_states] = -ca.inf
        lbx[1: n_states * (self.__N + 1): n_states] = -ca.inf
        lbx[2: n_states * (self.__N + 1): n_states] = -ca.inf

        ubx[0: n_states * (self.__N + 1): n_states] = ca.inf
        ubx[1: n_states * (self.__N + 1): n_states] = ca.inf
        ubx[2: n_states * (self.__N + 1): n_states] = ca.inf

        lbx[n_states * (self.__N + 1):] = self.__omega_min
        ubx[n_states * (self.__N + 1):] = self.__omega_max

        self.__args = dict(lbg=ca.DM.zeros((n_states * (self.__N + 1), 1)), ubg=ca.DM.zeros((n_states * (self.__N + 1), 1)), lbx=lbx, ubx=ubx)


    def operate(self,cat_states=False, cat_controls=False, X0 = False):
        self.init_variables()
        self.set_rhs()
        self.obj_fin()
        self.sim_loop(self.__args,X0, self.__states.numel(), self.__controls.numel(), self.__pack, cat_states, cat_controls ,self.create_solver(self.__obj,self.__g, self.__pack[3],self.__states.numel(), self.__controls.nume(),self.__pack))






