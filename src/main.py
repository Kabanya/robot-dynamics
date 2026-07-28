import pinocchio as pin
import example_robot_data
import meshcat.geometry as g
from pinocchio.visualize import MeshcatVisualizer

from src.com import CoMClass
from src.footsteps import FootSteps
from src.legs import LeftLeg, RightLeg
from src.plotting import Plotter
from src.talos import Talos
from src.zmp import ZmpClass

robot_com = 0.8766814
DT = 0.1 # 0.05

robot = example_robot_data.load("talos")
model = robot.model
data = robot.data

footsteps = FootSteps([.0, -.1], [.0, .1])
footsteps.add_phase(.3, 'none')
footsteps.add_phase(.7, 'left', [.1, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.2, -.1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'left', [.3, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.4, -.1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'left', [.5, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.5, -.1])
footsteps.add_phase(.5, 'none')

print("Computing started...")
zmp_traj = ZmpClass(footsteps)
com_traj = CoMClass(zmp_traj, com_z_nominal=robot_com)

# Initialize visualizer
robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(open=True)
robot.loadViewerModel("pinocchio")
robot.display(robot.q0)

# Visualize CoM and ZMP
meshcat_viz = robot.viz.viewer
meshcat_viz["zmp"].set_object(g.Sphere(0.050), g.MeshLambertMaterial(color=0xff0000))  # red - ZMP
meshcat_viz["com"].set_object(g.Sphere(0.125), g.MeshLambertMaterial(color=0x0000ff))  # blue - CoM

pin.framesForwardKinematics(model, data, robot.q0)
# set foots
left_foot_id = model.getFrameId("leg_left_sole_fix_joint")
right_foot_id = model.getFrameId("leg_right_sole_fix_joint")
# set initial foot positions
initial_left_height = data.oMf[left_foot_id].translation[2]
initial_right_height = data.oMf[right_foot_id].translation[2]
base_foot_height = (initial_left_height + initial_right_height) / 2.0
left_ank = LeftLeg(footsteps, base_foot_height)
right_ank = RightLeg(footsteps, base_foot_height)
# Calculate total time for the animation
t_total = footsteps.timetime[-1]

animator = Talos(
    robot, model, data, left_foot_id, right_foot_id,
    left_ank, right_ank, com_traj, footsteps,
    robot_com, base_foot_height, DT,
    zmp_traj=zmp_traj
)
animator.animate(t_total)

Plotter.plot_time_series(com_traj, left_ank, right_ank, t_total)
Plotter.plot_zmp_com_feet(zmp_traj, com_traj, left_ank, right_ank, t_total, DT)
