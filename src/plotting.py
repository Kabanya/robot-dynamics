import matplotlib.pyplot as plt
import numpy as np

from .legs import LeftLeg, RightLeg

class Plotter:
    @staticmethod
    def _prepare_trajectories(com_traj, left_ank, right_ank, t_total, num_points=100):
        t_plot = np.linspace(0, t_total, num_points)
        com_traj_plot = np.array([com_traj(t) for t in t_plot])
        left_foot_traj = np.array([left_ank(t) for t in t_plot])
        right_foot_traj = np.array([right_ank(t) for t in t_plot])
        return t_plot, com_traj_plot, left_foot_traj, right_foot_traj

    @staticmethod
    def plot_x_trajectories(t_plot, com_traj_plot, left_foot_traj, right_foot_traj):
        plt.plot(t_plot, com_traj_plot[:, 0], label='CoM X')
        plt.plot(t_plot, left_foot_traj[:, 0], '--', label='Left foot X')
        plt.plot(t_plot, right_foot_traj[:, 0], '--', label='Right foot X')
        plt.title('CoM X trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('X position (m)')
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_y_trajectories(t_plot, com_traj_plot, left_foot_traj, right_foot_traj):
        plt.plot(t_plot, com_traj_plot[:, 1], label='CoM Y')
        plt.plot(t_plot, left_foot_traj[:, 1], '--', label='Left foot Y')
        plt.plot(t_plot, right_foot_traj[:, 1], '--', label='Right foot Y')
        plt.title('CoM Y trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Y position (m)')
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_xy_trajectories(com_traj_plot, left_foot_traj, right_foot_traj):
        plt.plot(com_traj_plot[:, 0], com_traj_plot[:, 1], label='CoM XY')
        plt.plot(left_foot_traj[:, 0], left_foot_traj[:, 1], '--', label='Left foot XY')
        plt.plot(right_foot_traj[:, 0], right_foot_traj[:, 1], '--', label='Right foot XY')
        plt.title('CoM XY trajectory')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

    @staticmethod
    def plot_time_series(com_traj, left_ank, right_ank, t_total):
        t_plot, com_traj_plot, left_foot_traj, right_foot_traj = Plotter._prepare_trajectories(
            com_traj, left_ank, right_ank, t_total
        )
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        Plotter.plot_x_trajectories(t_plot, com_traj_plot, left_foot_traj, right_foot_traj)

        plt.subplot(1, 3, 2)
        Plotter.plot_y_trajectories(t_plot, com_traj_plot, left_foot_traj, right_foot_traj)

        plt.subplot(1, 3, 3)
        Plotter.plot_xy_trajectories(com_traj_plot, left_foot_traj, right_foot_traj)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_zmp_com_feet(zmptraj, com_traj, left_ank, right_ank, t_total, dt=0.05):
        N = int(t_total / dt)
        ts = np.linspace(0, t_total, N)

        zmp_points = np.array([zmptraj(t) for t in ts])
        com_points = np.array([com_traj(t) for t in ts])
        left_points = np.array([left_ank(t) for t in ts])
        right_points = np.array([right_ank(t) for t in ts])

        plt.figure(figsize=(8, 6))
        plt.plot(zmp_points[:, 0], zmp_points[:, 1], label='ZMP Trajectory', color='red')
        plt.plot(com_points[:, 0], com_points[:, 1], label='CoM Trajectory', color='blue')
        plt.plot(left_points[:, 0], left_points[:, 1], '--', label='Left Foot', color='green')
        plt.plot(right_points[:, 0], right_points[:, 1], '--', label='Right Foot', color='orange')

        plt.legend()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Trajectories: ZMP, CoM, Feet')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_zmp(footsteps):
        from .zmp import ZmpClass
        test = ZmpClass(footsteps)
        t_max = footsteps.timetime[-1]
        dt = 0.1
        N = int(t_max / dt + 1)
        zmp_traj = np.zeros((N, 2))
        left_traj = np.zeros((N, 2))
        right_traj = np.zeros((N, 2))

        for k in range(N):
            t = k * dt
            zmp_traj[k, :] = test(t)
            left_traj[k, :] = footsteps.get_left_position(t)
            right_traj[k, :] = footsteps.get_right_position(t)

        plt.figure(figsize=(10, 8))
        plt.plot(zmp_traj[:, 0], zmp_traj[:, 1], 'ro-', label='ZMP trajectory', markersize=3)
        plt.plot(left_traj[:, 0], left_traj[:, 1], 'bx-', label='Left foot', markersize=4)
        plt.plot(right_traj[:, 0], right_traj[:, 1], 'gx-', label='Right foot', markersize=4)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.legend()
        plt.grid(True)
        plt.title('ZMP and Foot Trajectories')
        plt.show()

    @staticmethod
    def plot_footsteps(footsteps):
        Plotter.plot_zmp(footsteps)

    @staticmethod
    def plot_all_trajectories(com_traj, left_ank, right_ank, t_total, footsteps):
        Plotter.plot_time_series(com_traj, left_ank, right_ank, t_total)
        Plotter.plot_footsteps(footsteps)
        print("All trajectory plots generated successfully!")

    @staticmethod
    def plot_ctrl_error_x(com_traj):
        if com_traj.solver != "LCQP":
            return

        ref_x= np.zeros((com_traj.N,1))
        act_u= com_traj.x_opt[2::3]
        act_c= com_traj.x_opt[0::3]
        act_dc=com_traj.x_opt[1::3]
        ddc= (act_dc[1:-1]-act_dc[0:-2])/com_traj.dt
        act_x=act_c-com_traj.L*act_u
        for k in range(com_traj.N):
            ref_x[k]=com_traj.zmp_traj(com_traj.dt*k)[0]

        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(ref_x, label='Reference ZMP X')
        plt.plot(act_x,'x', label='Actual ZMP X')
        plt.legend()
        plt.grid(True)
        plt.title('ZMP X Tracking')
        plt.subplot(1,2,2)
        plt.plot(act_u, label='Control Input')
        plt.plot(ddc,'x', label='Acceleration')
        plt.legend()
        plt.grid(True)
        plt.title('Control and Acceleration X')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ctrl_error_y(com_traj):
        if com_traj.solver != "LCQP":
            return

        ref_y= np.zeros((com_traj.N,1))
        act_u= com_traj.y_opt[2::3]
        act_c= com_traj.y_opt[0::3]
        act_dc=com_traj.y_opt[1::3]
        ddc= (act_dc[1:-1]-act_dc[0:-2])/com_traj.dt
        act_y=act_c-com_traj.L*act_u
        for k in range(com_traj.N):
            ref_y[k]=com_traj.zmp_traj(com_traj.dt*k)[1]

        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(ref_y, label='Reference ZMP Y')
        plt.plot(act_y,'x', label='Actual ZMP Y')
        plt.legend()
        plt.grid(True)
        plt.title('ZMP Y Tracking')
        plt.subplot(1,2,2)
        plt.plot(act_u, label='Control Input')
        plt.plot(ddc,'x', label='Acceleration')
        plt.legend()
        plt.grid(True)
        plt.title('Control and Acceleration Y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_com_traj(com_traj):
        plt.figure()
        plt.plot(com_traj.x_traj, com_traj.y_traj, label='CoM Trajectory')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title('CoM XY Trajectory')
        plt.show()


    @staticmethod
    def plot_legs(footsteps, base_foot_height=0.0):
        tmp_left = LeftLeg(footsteps, base_foot_height)
        tmp_right = RightLeg(footsteps, base_foot_height)

        t_max = footsteps.timetime[-1]
        dt = 0.01
        N = int(t_max / dt + 1)
        ankle_left_traj = np.zeros((N, 3))
        ankle_right_traj = np.zeros((N, 3))
        for k in range(N):
            t_q = dt * k
            ankle_left_traj[k, :] = tmp_left(t_q)
            ankle_right_traj[k, :] = tmp_right(t_q)
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(ankle_left_traj[:, 0], ankle_left_traj[:, 1], ankle_left_traj[:, 2], label='Left leg')
        ax.plot(ankle_right_traj[:, 0], ankle_right_traj[:, 1], ankle_right_traj[:, 2], label='Right leg')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
