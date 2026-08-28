import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

robot_d1 = 0.475
robot_a1 = 0
robot_alpha1 = np.pi/2

robot_d2 = 0
robot_a2 = 0.88
robot_alpha2 = 0

robot_d3 = 0
robot_a3 = 0.2
robot_alpha3 = np.pi/2

robot_d4 = 0.82
robot_a4 = 0
robot_alpha4 = -np.pi/2

robot_d5 = 0
robot_a5 = 0
robot_alpha5 = np.pi/2

robot_d6 = 0.14
robot_a6 = 0
robot_alpha6 = 0

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
    return sm.SE3(T)

def printTransformationMatrixForPos(theta1, theta2, theta3, theta4, theta5, theta6):
    """
    Print the transformation matrix for a given set of joint angles.

    Parameters:
    theta1 (float): Joint angle for joint 1 in radians.
    theta2 (float): Joint angle for joint 2 in radians.
    theta3 (float): Joint angle for joint 3 in radians.
    theta4 (float): Joint angle for joint 4 in radians.
    theta5 (float): Joint angle for joint 5 in radians.
    theta6 (float): Joint angle for joint 6 in radians.
    """
    T10 = getTransformationMatrix(theta1, robot_d1, robot_a1, robot_alpha1)
    T21 = getTransformationMatrix(theta2, robot_d2, robot_a2, robot_alpha2)
    T32 = getTransformationMatrix(theta3, robot_d3, robot_a3, robot_alpha3)
    T43 = getTransformationMatrix(theta4, robot_d4, robot_a4, robot_alpha4)
    T54 = getTransformationMatrix(theta5, robot_d5, robot_a5, robot_alpha5)
    T65 = getTransformationMatrix(theta6, robot_d6, robot_a6, robot_alpha6)
    
    T60 = T10 @ T21 @ T32 @ T43 @ T54 @ T65
    
    print("T10")
    print(T10)
    print("T21")
    print(T21)
    print("T32")
    print(T32)
    print("T43")
    print(T43)
    print("T54")
    print(T54)
    print("T65")
    print(T65)
    print("T60")
    print(T60)
    print("--------------------------------------------------------------------")

printTransformationMatrixForPos(np.pi/2, 0, 0, 0, 0, 0)
printTransformationMatrixForPos(0, 0, 0, 0, 0, 0)
printTransformationMatrixForPos(np.pi/4, np.pi/4, -np.pi/2, np.pi/3, 0, 0)


def visualizeUsingPeterCorke(theta1, theta2, theta3, theta4, theta5, theta6):
    """
    Visualize the robot configuration using Peter Corke's Robotics Toolbox.

    Parameters:
    theta1 (float): Joint angle for joint 1 in radians.
    theta2 (float): Joint angle for joint 2 in radians.
    theta3 (float): Joint angle for joint 3 in radians.
    theta4 (float): Joint angle for joint 4 in radians.
    theta5 (float): Joint angle for joint 5 in radians.
    theta6 (float): Joint angle for joint 6 in radians.
    """
    # Define the robot using standard DH parameters
    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=robot_d1, a=robot_a1, alpha=robot_alpha1),
        rtb.RevoluteDH(d=robot_d2, a=robot_a2, alpha=robot_alpha2),
        rtb.RevoluteDH(d=robot_d3, a=robot_a3, alpha=robot_alpha3),
        rtb.RevoluteDH(d=robot_d4, a=robot_a4, alpha=robot_alpha4),
        rtb.RevoluteDH(d=robot_d5, a=robot_a5, alpha=robot_alpha5),
        rtb.RevoluteDH(d=robot_d6, a=robot_a6, alpha=robot_alpha6)
    ], name="6-DOF Robot")

    # Create the joint configuration vector
    q = [theta1, theta2, theta3, theta4, theta5, theta6]

    # Plot the robot configuration
    robot.plot(q, block=True)


visualizeUsingPeterCorke(np.pi/2, 0, 0, 0, 0, 0)
visualizeUsingPeterCorke(0, 0, 0, 0, 0, 0)
visualizeUsingPeterCorke(np.pi/4, np.pi/4, -np.pi/2, np.pi/3, 0, 0)

def calculateInverseKinematics(px, py, pz, roll, pitch, yaw):
    """
    Calculate the inverse kinematics for a 6-DOF robot given the desired end-effector position and orientation.

    Parameters:
    px (float): Desired x-coordinate of the end-effector.
    py (float): Desired y-coordinate of the end-effector.
    pz (float): Desired z-coordinate of the end-effector.
    roll (float): Desired roll angle in radians.
    pitch (float): Desired pitch angle in radians.
    yaw (float): Desired yaw angle in radians.

    Returns:
    list: Joint angles [theta1, theta2, theta3, theta4, theta5, theta6] in radians.
    """
    # Create the desired transformation matrix from position and orientation
    T_desired = sm.SE3(px, py, pz) * sm.SE3.RPY(roll, pitch, yaw)

    # Define the robot using standard DH parameters
    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=robot_d1, a=robot_a1, alpha=robot_alpha1),
        rtb.RevoluteDH(d=robot_d2, a=robot_a2, alpha=robot_alpha2),
        rtb.RevoluteDH(d=robot_d3, a=robot_a3, alpha=robot_alpha3),
        rtb.RevoluteDH(d=robot_d4, a=robot_a4, alpha=robot_alpha4),
        rtb.RevoluteDH(d=robot_d5, a=robot_a5, alpha=robot_alpha5),
        rtb.RevoluteDH(d=robot_d6, a=robot_a6, alpha=robot_alpha6)
    ], name="6-DOF Robot")

    # Calculate inverse kinematics using the toolbox
    q_solution = robot.ikine_LM(T_desired)

    if q_solution.success:
        print(f"Inverse Kinematics Solution Found: {q_solution.q}")
        return q_solution.q
    else:
        print("Error: Inverse kinematics solution not found.")
        return None

calculateInverseKinematics(0.5, 0.2, 0.3, np.pi/4, np.pi/6, np.pi/3)
calculateInverseKinematics(0.8, 0.1, 0.5, np.pi/2, np.pi/4, np.pi/6)
calculateInverseKinematics(0.06, 0.06, 0.277, 0, 0, 0)