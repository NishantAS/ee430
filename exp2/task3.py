import math
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

def get_transformation_matrix(a, alpha, d, theta):
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                  np.cos(alpha),                 d],
        [0,              0,                              0,                             1]
    ])
    return T

def jacobian_n_link_planar(dh, joint_type):
    n, _ = dh.shape
    J = np.zeros((6, n))
    if len(joint_type) != n:
        print("error")
        return
    T = np.zeros((n+1, 4, 4))
    T[0] = np.eye(4)
    for i in range(n):
        T[i+1] = T[i] @ get_transformation_matrix(*(dh[i]))

    for i, t in enumerate(joint_type):
        match t:
            case 'r':
                J[:3, i] = np.cross(T[i, :3, 2], T[-1, :3, 3] - T[i, :3, 3]) 
                J[3:, i] = T[i, :3, 2]
            case 'p':
                J[:3, i] = T[i, :3, 2]

    return J

# Tolerance and No of iterations
tol = 0.001
iterations = 209

q = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi], dtype=float)
mutarget = np.array([5, 5, 5, np.pi, np.pi, np.pi])

robot = rtb.DHRobot([
    rtb.RevoluteDH(a=0, alpha=np.pi/2, d=0.475),
    rtb.RevoluteDH(a=0.88, alpha=0, d=0),
    rtb.RevoluteDH(a=0.2, alpha=np.pi/2, d=0),
    rtb.RevoluteDH(a=0, alpha=-np.pi/2, d=0.82),
    rtb.RevoluteDH(a=0, alpha=np.pi/2, d=0),
    rtb.RevoluteDH(a=0, alpha=0, d=0.14)
], name="6-DOF Robot")

# Newton-Raphson Iteration
for i in range(iterations):
    # Joint positions calculation
    T = robot.fkine(q)
    muguess = np.array([T.t[0], T.t[1], T.t[2], T.rpy()[0], T.rpy()[1], T.rpy()[2]])

    # robot.plot(q)

    # Plotting current robot configuration
    delta = mutarget - muguess


    if np.linalg.norm(delta) < tol:
        print(f"\nConverged in {i} iterations!")
        print(f"Final Joint Angles: q1 = {math.degrees(q[0]):.2f}°, q2 = {math.degrees(q[1]):.2f}° q3 = {math.degrees(q[2]):.2f}° q4 = {math.degrees(q[3]):.2f}° q5 = {math.degrees(q[4]):.2f}° q6 = {math.degrees(q[5]):.2f}°")
        break

    jacobian = jacobian_n_link_planar(np.array([
            [link.a, link.alpha, link.d, q[i]]
            for i, link in enumerate(robot.links)
        ]), 'rrrrrr')

    # Newton Update
    q += np.linalg.pinv(jacobian) @ delta

    q = np.arctan2(np.sin(q), np.cos(q))
else:
    print(f"\nDid not converge in {iterations} iterations.")
    print(f"Final Joint Angles: q1 = {math.degrees(q[0]):.2f}°, q2 = {math.degrees(q[1]):.2f}°")


robot.plot(q, block=True)
