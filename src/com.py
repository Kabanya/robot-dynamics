## Center of Mass

import numpy as np
from pinocchio.utils import eye, zero

from .factor import Factor, FactorGraph
from .zmp import ZmpClass

MASS = 10.0

class CoMClass(object):
    def __init__(self, zmp_traj: ZmpClass,
                 com_z_nominal: float = 0.9,
                 solver='LCQP',
                 flag_plot=False):

        self.zmp_traj = zmp_traj
        self.dt=0.005 #sampling time
        self.t_end= self.zmp_traj.footsteps.timetime[-1]
        self.com_z=com_z_nominal
        self.g=9.8
        self.x_opt=[]
        self.y_opt=[]
        self.N=int(self.t_end/self.dt)
        self.L=self.com_z/self.g
        self.solver=solver
        self.flag_plot=flag_plot

        self.x_traj = self.solve_CoM_traj(0) #solve x trajectory
        self.y_traj = self.solve_CoM_traj(1) #y

    def solve_CoM_traj(self, dir: int):
        '''
        dir: 0 for x direction, 1 for y direction
        Solves CoM trajectory ensuring ZMP = CoM projection (zero moment).
        '''
        direction_name = "X" if dir == 0 else "Y"

        N = self.N
        nx = 3
        L = self.L  # com_z / g
        f = FactorGraph(nx, N)

        M1=np.array([[1, self.dt, 0], [0, 1, self.dt], [0, 0, 1]])
        M2=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        constraint_count = 0
        for k in range(N-1):
            f.add_factor_constraint([Factor(k, M1), Factor(k+1, -M2)], zero(3))
            constraint_count += 1

        Mc = np.array([[1, 0, -L]])
        Mf=eye(1)
        zmp_constraint_count = 0
        zmp_refs = []

        for k in range(N):
            traj_x = self.zmp_traj(self.dt * k)[dir]
            zmp_refs.append(traj_x)
            f.add_factor_constraint([Factor(k, Mc)], Mf * traj_x)
            zmp_constraint_count += 1
        opt = f.solve() # LCQP solver

        if dir == 0:
            self.x_opt=np.copy(opt)
        else:
            self.y_opt=np.copy(opt)

        trajectory = np.reshape(opt[0::3], len(opt[0::3]))

        return trajectory

    def __call__(self, t):
        idx = int(t / self.dt)
        idx = min(idx, len(self.x_traj) - 1) # check bounds
        idx = max(idx, 0)
        result = np.array([self.x_traj[idx], self.y_traj[idx], self.com_z])
        return result

    def compute_moment(self, t):
        idx = int(t / self.dt)
        idx = min(idx, self.N - 1)
        idx = max(idx, 0)

        com_pos_x = self.x_traj[idx]
        com_pos_y = self.y_traj[idx]

        ddc_x = self.x_opt[idx * 3 + 2]
        ddc_y = self.y_opt[idx * 3 + 2]

        zmp_ref = self.zmp_traj(t) # This is [zmp_ref_x, zmp_ref_y]
        moment_y_component = MASS * self.g * (com_pos_x - zmp_ref[0] - self.L * ddc_x)
        moment_x_component = -MASS * self.g * (com_pos_y - zmp_ref[1] - self.L * ddc_y)

        moment_norm = np.linalg.norm([moment_x_component, moment_y_component])
        return moment_norm
