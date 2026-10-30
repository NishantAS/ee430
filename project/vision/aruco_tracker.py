"""
ArUco Paper Localization & Target Quadrant Pose Estimation (Criterion B1, B3).

Detects corner ArUco markers (DICT_4X4_50, IDs 0, 1, 2, 3), estimates the 6-DOF pose
of the Paper Frame {M} using cv2.solvePnP / multi-marker board estimation with CLAHE preprocessing,
and transforms target quadrant coordinates into Robot Base Coordinates:
    ^0 p_target = ^0 T_6 * ^6 T_C * ^C T_M * ^M p_target
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.aruco_board import QUADRANT_CENTERS_MM, MARKER_SIZE_MM, MARKER_POSITIONS_MM
from kinematics.dh_model import forward_kinematics_spong, is_target_reachable


class ArUcoTracker:
    def __init__(self, k_matrix: np.ndarray, dist_coeffs: np.ndarray, t_flange_to_cam: np.ndarray):
        self.K = k_matrix
        self.D = dist_coeffs
        self.T_6_C = t_flange_to_cam

        try:
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            self.use_modern_api = True
        except AttributeError:
            self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_modern_api = False

        # CLAHE enhancer to suppress specular reflection / harsh lab lighting
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 3D corner coordinates for each marker in paper frame {M}
        self.marker_3d_corners = self._build_marker_3d_corners()

    def _build_marker_3d_corners(self) -> Dict[int, np.ndarray]:
        """Builds 3D corner coordinates for each marker ID in paper frame {M}."""
        s2 = MARKER_SIZE_MM / 2.0
        corners_3d = {}
        for m_id, center in MARKER_POSITIONS_MM.items():
            cx, cy = center[0], center[1]
            # Corners in standard order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            corners_3d[m_id] = np.array([
                [cx - s2, cy + s2, 0.0],
                [cx + s2, cy + s2, 0.0],
                [cx + s2, cy - s2, 0.0],
                [cx - s2, cy - s2, 0.0],
            ], dtype=np.float32)
        return corners_3d

    def detect_markers(self, frame: np.ndarray):
        """Preprocesses frame with CLAHE and detects ArUco markers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)

        if self.use_modern_api:
            corners, ids, rejected = self.detector.detectMarkers(enhanced_gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(enhanced_gray, self.dictionary, parameters=self.detector_params)

        return corners, ids

    def estimate_paper_pose(self, corners, ids) -> Optional[np.ndarray]:
        """
        Computes 4x4 homogeneous transform ^C T_M (Paper Frame {M} in Camera Frame {C})
        using all detected markers via cv2.solvePnP.
        """
        if ids is None or len(ids) == 0:
            return None

        obj_pts_list = []
        img_pts_list = []

        ids_flat = ids.flatten()
        for i, m_id in enumerate(ids_flat):
            if m_id in self.marker_3d_corners:
                obj_pts_list.append(self.marker_3d_corners[m_id])
                img_pts_list.append(corners[i][0])

        if len(obj_pts_list) == 0:
            return None

        all_obj_pts = np.vstack(obj_pts_list).astype(np.float32)
        all_img_pts = np.vstack(img_pts_list).astype(np.float32)

        # Multi-point PnP
        success, rvec, tvec = cv2.solvePnP(
            all_obj_pts, all_img_pts, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Build 4x4 ^C T_M
        R, _ = cv2.Rodrigues(rvec)
        t_c_m = np.eye(4, dtype=np.float64)
        t_c_m[0:3, 0:3] = R
        t_c_m[0:3, 3] = tvec.flatten()
        return t_c_m

    def compute_quadrant_world_target(
        self,
        quadrant_id: int,
        t_cam_paper: np.ndarray,
        current_joint_angles_deg: List[float],
        hover_height_z: float = 100.0
    ) -> Tuple[Optional[List[float]], str]:
        """
        Transforms quadrant center into Base Frame {0}:
            ^0 p_quadrant = ^0 T_6(q_search) * ^6 T_C * ^C T_M * ^M p_quadrant

        Then computes camera-centered hover command:
            [X_cmd, Y_cmd, Z_hover, Rx, Ry, Rz]
        """
        if quadrant_id not in QUADRANT_CENTERS_MM:
            return None, f"Invalid quadrant id: {quadrant_id}"

        # 1. Forward Kinematics of arm at search pose: ^0 T_6
        t_0_6 = forward_kinematics_spong(current_joint_angles_deg)

        # 2. Quadrant center in Paper Frame {M}
        p_m = np.append(QUADRANT_CENTERS_MM[quadrant_id]["pos"], 1.0)  # homogeneous [x, y, z, 1]

        # 3. Overall transformation chain:
        # ^0 p = ^0 T_6 * ^6 T_C * ^C T_M * ^M p
        p_world_target = t_0_6 @ self.T_6_C @ t_cam_paper @ p_m
        tx, ty = p_world_target[0], p_world_target[1]

        # 4. Subtract camera lens offset so camera optical axis points directly at target
        # For top-down vertical hover (Rx=180, Ry=0, Rz=-90)
        # ^6 T_C horizontal offset along X6 is -30 mm
        flange_target_x = tx - self.T_6_C[0, 3]
        flange_target_y = ty - self.T_6_C[1, 3]
        flange_target_z = hover_height_z

        # 5. Check workspace safety
        is_safe, reason = is_target_reachable(flange_target_x, flange_target_y, flange_target_z)
        if not is_safe:
            return None, f"Target out of safe workspace: {reason}"

        # Commanded coords: [X, Y, Z, Rx, Ry, Rz]
        target_coords = [
            float(flange_target_x),
            float(flange_target_y),
            float(flange_target_z),
            180.0,
            0.0,
            -90.0
        ]
        return target_coords, "Target pose successfully computed."

    def draw_detections(self, frame: np.ndarray, corners, ids, t_cam_paper: Optional[np.ndarray] = None) -> np.ndarray:
        """Visualizes detected markers, board axes, and quadrant projections on the frame."""
        vis = frame.copy()
        if ids is not None and len(ids) > 0:
            if self.use_modern_api:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            else:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        if t_cam_paper is not None:
            rvec, _ = cv2.Rodrigues(t_cam_paper[0:3, 0:3])
            tvec = t_cam_paper[0:3, 3]
            try:
                cv2.drawFrameAxes(vis, self.K, self.D, rvec, tvec, 50.0, 3)
            except AttributeError:
                pass

            # Project quadrant centers into image
            for q_id, q_info in QUADRANT_CENTERS_MM.items():
                p_3d = np.array([q_info["pos"]], dtype=np.float32)
                img_pts, _ = cv2.projectPoints(p_3d, rvec, tvec, self.K, self.D)
                pt = tuple(img_pts[0].ravel().astype(int))
                cv2.circle(vis, pt, 6, (0, 0, 255), -1)
                cv2.putText(vis, f"Q{q_id}", (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis

