"""
Spong Standard Denavit-Hartenberg (DH) Kinematics Model for myCobot 280 Jetson Nano.

Reference:
- Spong, Hutchinson & Vidyasagar, "Robot Modeling and Control", Chapter 3.
- myCobot 280 hardware specification & link parameters.

Spong Standard DH transformation definition from frame {i-1} to frame {i}:
    A_i = Rot(z, theta_i) * Trans(z, d_i) * Trans(x, a_i) * Rot(x, alpha_i)

    A_i = [
        [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0,           sin(alpha),             cos(alpha),            d           ],
        [0,           0,                      0,                     1           ]
    ]

Link lengths (in millimeters):
    d1 = 131.56 mm  (Base to J2 elevation)
    a2 = -110.40 mm (Upper arm link J2 to J3)
    a3 = -96.00 mm  (Forearm link J3 to J4)
    d4 = 64.62 mm   (Wrist 1 offset J4 to J5)
    d5 = 73.18 mm   (Wrist 2 offset J5 to J6)
    d6 = 48.60 mm   (Flange offset J6 to End-Effector / Flange)
"""

import numpy as np
from typing import List, Tuple, Union

# myCobot 280 Standard DH parameters
# Link geometry in mm
D1 = 131.56
A2 = -110.40
A3 = -96.00
D4 = 64.62
D5 = 73.18
D6 = 48.60

# Spong DH Table: [theta_offset (rad), d (mm), a (mm), alpha (rad)]
# Commanded joint angles q in radians are added to theta_offset
SPONG_DH_TABLE = [
    {"theta_offset": 0.0,         "d": D1,   "a": 0.0,   "alpha": np.pi / 2},
    {"theta_offset": -np.pi / 2,  "d": 0.0,  "a": A2,    "alpha": 0.0},
    {"theta_offset": 0.0,         "d": 0.0,  "a": A3,    "alpha": 0.0},
    {"theta_offset": -np.pi / 2,  "d": D4,   "a": 0.0,   "alpha": np.pi / 2},
    {"theta_offset": np.pi / 2,   "d": D5,   "a": 0.0,   "alpha": -np.pi / 2},
    {"theta_offset": 0.0,         "d": D6,   "a": 0.0,   "alpha": 0.0},
]


def dh_transform_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Computes standard Spong DH homogeneous transformation matrix A_i.

    Parameters:
        theta (float): Joint angle in radians (including offset).
        d (float): Link offset along z_{i-1} in mm.
        a (float): Link length along common normal x_i in mm.
        alpha (float): Link twist about common normal x_i in radians.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix A_i.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Clean numerical near-zero precision
    if abs(ca) < 1e-12:
        ca = 0.0
    if abs(sa) < 1e-12:
        sa = 0.0
    if abs(ct) < 1e-12:
        ct = 0.0
    if abs(st) < 1e-12:
        st = 0.0

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d     ],
        [0.0, 0.0,      0.0,     1.0   ]
    ], dtype=np.float64)


def forward_kinematics_spong(joint_angles_deg: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Computes the complete Forward Kinematics homogeneous transformation matrix ^0 T_6.

    Parameters:
        joint_angles_deg (list or array of 6 floats): Joint angles q1...q6 in DEGREES.

    Returns:
        np.ndarray: 4x4 transformation matrix ^0 T_6 mapping Frame {6} (Flange) to Frame {0} (Base).
    """
    if len(joint_angles_deg) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_deg)}")

    joint_angles_rad = np.radians(joint_angles_deg)
    t_total = np.eye(4, dtype=np.float64)

    for i in range(6):
        theta = joint_angles_rad[i] + SPONG_DH_TABLE[i]["theta_offset"]
        d = SPONG_DH_TABLE[i]["d"]
        a = SPONG_DH_TABLE[i]["a"]
        alpha = SPONG_DH_TABLE[i]["alpha"]

        a_i = dh_transform_matrix(theta, d, a, alpha)
        t_total = t_total @ a_i

    return t_total


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts ZYX Euler angles (intrinsic yaw-pitch-roll / extrinsic Rx, Ry, Rz)
    matching pymycobot.get_coords() convention:
    pymycobot coordinates format is [x, y, z, rx, ry, rz] in (mm, degrees).
    R = Rz(rz) * Ry(ry) * Rx(rx).

    Parameters:
        R (np.ndarray): 3x3 rotation matrix.

    Returns:
        Tuple[float, float, float]: (rx, ry, rz) in degrees.
    """
    # Checking singularity at cos(ry) == 0
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock singularity
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    return np.degrees(rx), np.degrees(ry), np.degrees(rz)


def euler_zyx_to_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Constructs a 3x3 rotation matrix from ZYX Euler angles (rx, ry, rz) in degrees.
    """
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]
    ])

    # R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def get_flange_pose(joint_angles_deg: Union[List[float], np.ndarray]) -> List[float]:
    """
    Computes Cartesian pose [x, y, z, rx, ry, rz] matching pymycobot.get_coords() format.

    Parameters:
        joint_angles_deg (list of 6 floats): joint angles in degrees.

    Returns:
        List[float]: [x, y, z, rx, ry, rz] where x, y, z are in mm and rx, ry, rz are in degrees.
    """
    t_0_6 = forward_kinematics_spong(joint_angles_deg)
    x, y, z = t_0_6[0:3, 3]
    R = t_0_6[0:3, 0:3]
    rx, ry, rz = rotation_matrix_to_euler_zyx(R)
    return [float(x), float(y), float(z), float(rx), float(ry), float(rz)]


def is_target_reachable(x: float, y: float, z: float, r_min: float = 110.0, r_max: float = 235.0, z_min: float = 70.0) -> Tuple[bool, str]:
    """
    Safety guardrail enforcing the reachable hover workspace of myCobot 280.
    Checks reach envelope before dispatching commands to mc.send_coords().

    Parameters:
        x (float): Target X in mm.
        y (float): Target Y in mm.
        z (float): Target Z in mm.
        r_min (float): Minimum radius to avoid base collision (default 110 mm).
        r_max (float): Maximum radius for vertical downward orientation (default 235 mm).
        z_min (float): Minimum safe height above lab table (default 70 mm).

    Returns:
        Tuple[bool, str]: (is_safe, reason_message)
    """
    r_xy = np.sqrt(x**2 + y**2)
    r_total = np.sqrt(x**2 + y**2 + (z - D1)**2)

    if z < z_min:
        return False, f"Target Z={z:.1f} mm is below safe height limit of {z_min:.1f} mm (risk of table collision)."

    if r_xy < r_min:
        return False, f"Target planar radius r_xy={r_xy:.1f} mm is inside inner base dead-zone ({r_min:.1f} mm)."

    if r_xy > r_max:
        return False, f"Target planar radius r_xy={r_xy:.1f} mm exceeds vertical orientation envelope ({r_max:.1f} mm)."

    if r_total > 280.0:
        return False, f"Target total distance from shoulder {r_total:.1f} mm exceeds physical reach (280.0 mm)."

    return True, "Target is within safe workspace envelope."


if __name__ == "__main__":
    # Quick self-test with zero position
    home_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T = forward_kinematics_spong(home_angles)
    coords = get_flange_pose(home_angles)
    print("=== DH Model Self-Test ===")
    print("Home Angles:", home_angles)
    print("Homogeneous Matrix ^0 T_6:\n", np.round(T, 3))
    print(f"Pose [X, Y, Z, Rx, Ry, Rz]: [{coords[0]:.2f} mm, {coords[1]:.2f} mm, {coords[2]:.2f} mm, {coords[3]:.2f}°, {coords[4]:.2f}°, {coords[5]:.2f}°]")
    is_safe, reason = is_target_reachable(coords[0], coords[1], coords[2])
    print(f"Reachable check: {is_safe} ({reason})")

