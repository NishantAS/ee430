"""
Calibration data storage & Hand-Eye kinematics transformation utilities.

Contains:
- Camera intrinsic matrix K and distortion coefficients D.
- Eye-in-Hand transformation matrix ^6 T_C (from flange frame {6} to camera optical frame {C}).
- Coordinate frame conversion helpers for target quadrant calculation:
    ^0 p_target = ^0 T_6 * ^6 T_C * ^C T_M * ^M p_target
"""

import os
import numpy as np
from typing import Tuple, Optional

# Default Flange-to-Camera Offset for Elephant Robotics USB Camera
# Lens center relative to Flange {6}:
# Radial offset dx ~ -30.0 mm along flange housing, dz ~ +25.0 mm forward, dy ~ 0.0 mm
DEFAULT_FLANGE_TO_CAM_TRANSLATION = np.array([-30.0, 0.0, 25.0])  # in mm

# Camera optical frame: X right, Y down, Z forward
# Flange frame {6}: X out from flange, Y lateral, Z perpendicular to face
DEFAULT_FLANGE_TO_CAM_ROTATION = np.array([
    [0.0, -1.0,  0.0],
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0]
])

# Default Camera Intrinsics for 640x480 resolution (Flange USB Camera standard FOV ~60 deg)
DEFAULT_K = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)


def get_default_hand_eye_transform() -> np.ndarray:
    """Returns the 4x4 homogeneous transformation matrix ^6 T_C."""
    t_6_c = np.eye(4, dtype=np.float64)
    t_6_c[0:3, 0:3] = DEFAULT_FLANGE_TO_CAM_ROTATION
    t_6_c[0:3, 3] = DEFAULT_FLANGE_TO_CAM_TRANSLATION
    return t_6_c


def load_camera_calibration(calib_file: str = "project/calibration/camera_params.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads intrinsic matrix K, distortion coefficients dist_coeffs,
    and eye-in-hand matrix T_6_C from .npz file. If file not found, returns calibrated defaults.
    """
    if os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            k = data["K"]
            dist = data["dist_coeffs"]
            t_6_c = data["T_6_C"] if "T_6_C" in data else get_default_hand_eye_transform()
            return k, dist, t_6_c
        except Exception as e:
            print(f"Warning: Failed reading {calib_file} ({e}). Using default calibration.")

    return DEFAULT_K, DEFAULT_DIST_COEFFS, get_default_hand_eye_transform()


def save_camera_calibration(calib_file: str, k: np.ndarray, dist_coeffs: np.ndarray, t_6_c: np.ndarray):
    """Saves calibration parameters to .npz file."""
    os.makedirs(os.path.dirname(calib_file), exist_ok=True)
    np.savez(calib_file, K=k, dist_coeffs=dist_coeffs, T_6_C=t_6_c)
    print(f"Saved camera calibration parameters to: {calib_file}")

