import numpy as np
import roboticstoolbox as rtb


def calculate_rr_ik(px, py, a1, a2):
    """
    Calculates the Inverse Kinematics for a 2-link RR planar manipulator.
    """
    # Calculate D (cos(theta2))
    D = (px**2 + py**2 - a1**2 - a2**2) / (2 * a1 * a2)
    
    # Check if the point is reachable
    if abs(D) > 1:
        print(f"Error: The point ({px}, {py}) is not reachable by the manipulator.")
        return None
    
    # Calculate theta2 (using the elbow-down solution for this example)
    theta2 = np.arctan2(np.sqrt(1 - D**2), D)
    
    # Calculate theta1
    theta1 = np.arctan2(py, px) - np.arctan2(a2 * np.sin(theta2), a1 + a2 * np.cos(theta2))
    
    print(f"Success: Point is reachable.")
    print(f"Calculated Angles -> Theta 1: {np.degrees(theta1):.2f}°, Theta 2: {np.degrees(theta2):.2f}°")
    
    return [theta1, theta2]

# Example parameters
link1_length = 1.0
link2_length = 1.0
target_x = 1.5
target_y = 0.5

q_calculated = calculate_rr_ik(target_x, target_y, link1_length, link2_length)

def verify_with_toolbox(q, a1, a2, target_x, target_y):
    """
    Verifies the calculated joint angles using Peter Corke's Robotics Toolbox.
    """
    if q is None:
        print("Cannot verify: No valid joint angles provided.")
        return
        
    
    # Define the robot using standard DH parameters
    # Revolute joints, with link lengths a1 and a2
    robot = rtb.DHRobot([
        rtb.RevoluteDH(a=a1),
        rtb.RevoluteDH(a=a2)
    ], name="2-Link RR")
    
    # Calculate forward kinematics using the toolbox
    T_fk = robot.fkine(q)
    
    # Extract the resulting X and Y coordinates from the transformation matrix
    achieved_x = T_fk.t[0]
    achieved_y = T_fk.t[1]
    
    print(f"Target Point:   X = {target_x:.4f}, Y = {target_y:.4f}")
    print(f"Achieved Point: X = {achieved_x:.4f}, Y = {achieved_y:.4f}")
    
    # Verify if they match (accounting for tiny floating-point inaccuracies)
    if np.isclose(target_x, achieved_x) and np.isclose(target_y, achieved_y):
        print("Verification Successful: The manipulator reaches the target point!")
    else:
        print("Verification Failed: The manipulator does not reach the exact target point.")

# Run the verification step
if q_calculated is not None:
    verify_with_toolbox(q_calculated, link1_length, link2_length, target_x, target_y)

# Example of an unreachable point:
calculate_rr_ik(3.0, 3.0, link1_length, link2_length)