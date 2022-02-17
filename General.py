from sys import path
path.append(r"/home/zeiros/mpc_double_model/Sim_code")
path.append(r"/home/zeiros/casadi-linux-py38-v3.5.5-64bit")
import casadi as ca
from casadi import *
import math
import numpy as np
import matplotlib.pyplot as plt
from time import time


def shift_timestep(step_horizon, t0, state_init, u, f):
        f_value = f(state_init, u[:, 0])
        next_state = ca.DM.full(state_init + (step_horizon * f_value))

        t0 = t0 + step_horizon
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


class MPC:
    def __init__(self, x_init, y_init, theta_init, x_target, y_target, theta_target,h,N):
        self.__x_init = x_init
        self.__y_init = y_init
        self.__theta_init = theta_init
        self.__x_target = x_target
        self.__y_target = y_target
        self.__theta_target = theta_target
        self.__h = h
        self.__N = N


    def generate_expressions(self,states ,controls, rhs):
        n_states = states.numel()
        n_controls = controls.numel()
        f = ca.Function('f', [states, controls], [rhs])
        X = ca.SX.sym('X', n_states, (self.__N+1))
        U = ca.SX.sym('U', n_controls, self.__N)
        P = ca.SX.sym('P', n_states + n_states)
        pack = [f, X, U, P]
        return pack



    
    def create_solver(self, obj, g, P, n_states, n_controls, pack):
        OPT_variables = ca.vertcat(ca.reshape(pack[1], n_states * (self.__N + 1), 1), ca.reshape(pack[2], n_controls * self.__N, 1))
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
        # f: objective eqn
        # x: optmization variables
        # g: constrains 1
        # p: output parameters (constrains 2)

        opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
            },
        'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        return solver

        
    
    def sim_loop(self, args, X0, n_states, n_controls, pack, cat_states, cat_controls,solver,switch_pos = 1000000):
        state_init = ca.DM([self.__x_init, self.__y_init, self.__theta_init])
        state_target = ca.DM([self.__x_target, self.__y_target, self.__theta_target])
        f = pack[0]
        
        t0 = 0
        t = ca.DM(t0)
        u0 = ca.DM.zeros((n_controls, self.__N))  # initial control
        if X0 is False:
            X0 = ca.repmat(state_init, 1, self.__N + 1)  # initial state full

        mpc_iter = 0
        if cat_states is False:
            cat_states = DM2Arr(X0)
        if cat_controls is False:
            cat_controls = DM2Arr(u0[:, 0])
        times = np.array([[0]])
        sim_time = 200
        main_loop = time()
        pos = 0
        while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * self.__h < sim_time):
            if switch_pos == pos:
                break
            t1 = time()
            args['p'] = ca.vertcat(
                state_init,  # current state
                state_target  # target state
            )
            # optimization variable current state
            args['x0'] = ca.vertcat(
                ca.reshape(X0, n_states * (self.__N + 1), 1),
                ca.reshape(u0, n_controls * self.__N, 1)
            )

            sol = solver(
                x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p']
            )

            u = ca.reshape(sol['x'][n_states * (self.__N + 1):], n_controls, self.__N)
            X0 = ca.reshape(sol['x'][: n_states * (self.__N + 1)], n_states, self.__N + 1)

            cat_states = np.dstack((
                cat_states,
                DM2Arr(X0)
            ))

            cat_controls = np.vstack((
                cat_controls,
                DM2Arr(u[:, 0])
            ))
            t = np.vstack((
                t,
                t0
            ))

            t0, state_init, u0 = shift_timestep(self.__h, t0, state_init, u, f)

            print(X0)
            pos = int(X0[1, 0])
            X0 = ca.horzcat(
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            )

            # xx ...
            t2 = time()
            print(mpc_iter)
            print(t2 - t1)
            times = np.vstack((
                times,
                t2 - t1
            ))

            mpc_iter = mpc_iter + 1

        main_loop_time = time()
        ss_error = ca.norm_2(state_init - state_target)
        self.X0 = X0
        self.cat_controls = cat_controls
        self.cat_states = cat_states
        print('\n\n')
        print('Total time: ', main_loop_time - main_loop)
        print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
        print('final error: ', ss_error)




    
