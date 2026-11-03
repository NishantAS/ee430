"""
Our own forward-kinematics model of the myCobot 280, used ONLY to verify the robot (criterion A2:
"FK verified against your own DH model, with an error table"). All motion uses pymycobot's built-in
kinematics (send_angles / send_coords).

Spong standard DH convention (Spong, Hutchinson & Vidyasagar, ch. 3):
    A_i = Rot(z, theta_i) Trans(z, d_i) Trans(x, a_i) Rot(x, alpha_i),   theta_i = q_i + offset_i
Lengths in mm, public angles in degrees.
"""

from typing import List, Sequence

import numpy as np

# Link geometry (mm)
D1, A2, A3, D4, D5, D6 = 131.56, -110.40, -96.00, 64.62, 73.18, 48.60

# Spong DH table: (theta offset [rad], d [mm], a [mm], alpha [rad])
DH_TABLE = [
    (0.0,         D1,  0.0, np.pi / 2),
    (-np.pi / 2,  0.0, A2,  0.0),
    (0.0,         0.0, A3,  0.0),
    (-np.pi / 2,  D4,  0.0, np.pi / 2),
    (np.pi / 2,   D5,  0.0, -np.pi / 2),
    (0.0,         D6,  0.0, 0.0),
]


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


NOMINAL_LENGTHS = (D1, A2, A3, D4, D5, D6)
LENGTH_NAMES = ("d1", "a2", "a3", "d4", "d5", "d6")


def fk(q_deg: Sequence[float], lengths: Sequence[float] = NOMINAL_LENGTHS) -> np.ndarray:
    """^0T_6 = A_1 A_2 ... A_6: flange pose in the base frame for joint angles q (deg).
    `lengths` = (d1, a2, a3, d4, d5, d6) in mm overrides the link lengths (angles/twists are fixed)."""
    d1, a2, a3, d4, d5, d6 = lengths
    da = [(d1, 0.0), (0.0, a2), (0.0, a3), (d4, 0.0), (d5, 0.0), (d6, 0.0)]
    t = np.eye(4)
    for qi, (off, _, _, alpha), (d, a) in zip(np.radians(q_deg), DH_TABLE, da):
        t = t @ dh_matrix(qi + off, d, a, alpha)
    return t


def identify_lengths(qs: Sequence[Sequence[float]], positions: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Least-squares link lengths from measured poses. For fixed joint angles the flange position is LINEAR in
    the six lengths: p(q) = sum_i L_i * v_i(q), where v_i(q) is the direction that length i is measured along
    (z_{i-1} for a d, x_i for an a). Stacking 3 equations per pose gives an ordinary least-squares problem.
    """
    rows, rhs = [], []
    for q, p in zip(qs, positions):
        # v_i(q) = position with only length i set to 1 (exact because p is linear in the lengths)
        rows.append(np.column_stack([fk(q, np.eye(6)[i])[:3, 3] for i in range(6)]))
        rhs.append(np.asarray(p, float))
    lengths, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return lengths


def euler_zyx(rot: np.ndarray):
    """R = Rz(rz) Ry(ry) Rx(rx) -> (rx, ry, rz) in degrees (pymycobot get_coords convention)."""
    sy = np.hypot(rot[0, 0], rot[1, 0])
    if sy > 1e-6:
        rx, ry, rz = np.arctan2(rot[2, 1], rot[2, 2]), np.arctan2(-rot[2, 0], sy), np.arctan2(rot[1, 0], rot[0, 0])
    else:  # gimbal lock
        rx, ry, rz = np.arctan2(-rot[1, 2], rot[1, 1]), np.arctan2(-rot[2, 0], sy), 0.0
    return tuple(float(v) for v in np.degrees([rx, ry, rz]))


def euler_zyx_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return r_z @ r_y @ r_x


def flange_coords(q_deg: Sequence[float], lengths: Sequence[float] = NOMINAL_LENGTHS) -> List[float]:
    """Our FK in get_coords() format: [x, y, z, rx, ry, rz]."""
    t = fk(q_deg, lengths)
    return [float(v) for v in t[:3, 3]] + list(euler_zyx(t[:3, :3]))


def rotation_error_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices (convention-independent orientation error)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r1.T @ r2) - 1.0) / 2.0, -1.0, 1.0))))
