import numpy as np
import roboticstoolbox as rtb

# Link lengths
a1 = 1.0  # link-1 length
a2 = 1.0  # link-2 length

# DH Table
#  a_i  | alpha_i  | d_i  | theta_i
#  a1        0        0       *
#  a2        0        0       *

# Joint angles in degrees
theta1 = 0.0  # Angle of Motor-1
theta2 = 0.0  # Angle of Motor-2

# Convert angles to radians and store as joint configuration vector q
q = np.array([np.deg2rad(theta1), np.deg2rad(theta2)])

# Define DH Links using standard Denavit-Hartenberg convention (RevoluteDH)
L1 = rtb.RevoluteDH(d=0, a=a1, alpha=0)  # DH Table - Row 1
L2 = rtb.RevoluteDH(d=0, a=a2, alpha=0)  # DH Table - Row 2

# Create the serial link robot manipulator
robot = rtb.DHRobot([L1, L2], name="RR Planar")

# Print joint angles
print(f"The angles theta_1 = {theta1:3.2f} theta_2 = {theta2:3.2f}")

# Compute forward kinematics (returns an SE3 pose object)
print("Transformation Matrix T_0^2:")
T = robot.fkine(q)
print(T)

# Plot the robot configuration (block=True keeps the matplotlib figure open)
robot.plot(q, block=True)
