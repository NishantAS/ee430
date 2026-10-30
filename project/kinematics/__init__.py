"""
Kinematics package initialization for EE430 Course Project.
Implements Spong Standard DH Forward Kinematics and coordinate utilities.
"""

from .dh_model import (
    dh_transform_matrix,
    forward_kinematics_spong,
    get_flange_pose,
    is_target_reachable,
    rotation_matrix_to_euler_zyx,
    euler_zyx_to_rotation_matrix,
)

__all__ = [
    "dh_transform_matrix",
    "forward_kinematics_spong",
    "get_flange_pose",
    "is_target_reachable",
    "rotation_matrix_to_euler_zyx",
    "euler_zyx_to_rotation_matrix",
]

