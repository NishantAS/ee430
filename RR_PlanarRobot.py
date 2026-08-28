import roboticstoolbox as rtb

# Create links using DH parameters
L1 = rtb.RevoluteDH(d=0, a=1.0, alpha=0)
L2 = rtb.RevoluteDH(d=0, a=1.0, alpha=0)

# Build the serial link manipulator
robot = rtb.DHRobot([L1, L2], name="RR_Planar_Robot")

# Display forward kinematics result
q = [0, 0]  # Joint angles in radians
T = robot.fkine(q)
print("\nTransformation Matrix T (Robotics Toolbox):")
print(T)

# Plot the 2 DOF robot
robot.plot(q, block=True)