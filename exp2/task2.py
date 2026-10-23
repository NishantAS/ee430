import numpy as np

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

DH = np.array([
    [0.40,  np.deg2rad(90),  0.30, np.deg2rad(30)],  # Joint 1 - R
    [0.60,  np.deg2rad(0),   0.00, np.deg2rad(45)],  # Joint 2 - R
    [0.00,  np.deg2rad(-90), 0.25, np.deg2rad(0)],   # Joint 3 - P
    [0.20,  np.deg2rad(90),  0.10, np.deg2rad(60)]   # Joint 4 - R
])

J = jacobian_n_link_planar(DH, "rrpr")
print(J)
