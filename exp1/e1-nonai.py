import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

def getTransformationMatrix(theta, d, a, alpha):
    """
    Compute the transformation matrix using the Denavit-Hartenberg convention.

    Parameters:
    theta (float): Joint angle in radians.
    d (float): Link offset.
    a (float): Link length.
    alpha (float): Link twist in radians.

    Returns:
    numpy.ndarray: 4x4 transformation matrix.
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                  np.cos(alpha),                 d],
        [0,              0,                              0,                             1]
    ])
    return T

# Link lengths
a1 = 1.0  # link-1 length
a2 = 1.0  # link-2 length

# DH Table
#  a_i  | alpha_i  | d_i  | theta_i
#  a1        0        0       *
#  a2        0        0       *

# Joint angles in degrees
theta1 = 90.0
theta2 = 90.0

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
MYT = sm.SE3(getTransformationMatrix(q[0], 0, a1, 0) @ getTransformationMatrix(q[1], 0, a2, 0))
print(MYT)
print(T)

# Plot the robot configuration (block=True keeps the matplotlib figure open)
robot.plot(q, block=True)
# robot.teach(q, block=True)  # Open a teaching interface for the robot

#value = robot.ik_NR(T, end=L2)  # Compute inverse kinematics using Newton-Raphson method
value = robot.ik_NR(sm.SE3(1, 0, 0), end = L2, ilimit = 100)
print(value)