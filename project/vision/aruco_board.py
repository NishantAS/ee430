"""
ArUco Target Board Layout & Printable Sheet Generator (Criterion B1, B3).

Generates a printable A4 Target Sheet containing:
1. Four ArUco markers (DICT_4X4_50, IDs 0, 1, 2, 3) placed symmetrically at (+-75 mm, +-75 mm).
   - Marker size: 28 mm x 28 mm.
2. Centered Active Grid: 120 mm x 120 mm square divided into four 60 mm x 60 mm quadrants:
   - Quadrant 1 (Top-Left):     Happy     😊  (-30 mm, +30 mm)
   - Quadrant 2 (Top-Right):    Angry     😠  (+30 mm, +30 mm)
   - Quadrant 3 (Bottom-Right): Surprised 😮  (+30 mm, -30 mm)
   - Quadrant 4 (Bottom-Left):  Neutral   😐  (-30 mm, -30 mm)

Outputs:
- project/assets/aruco_target_sheet_A4.png (High-resolution 300 DPI for direct printing)
"""

import os
import numpy as np

# Coordinates of quadrant centers in Paper Frame {M} in millimeters:
QUADRANT_CENTERS_MM = {
    1: {"name": "Happy",     "pos": np.array([-30.0,  30.0, 0.0])},
    2: {"name": "Angry",     "pos": np.array([ 30.0,  30.0, 0.0])},
    3: {"name": "Surprise",  "pos": np.array([ 30.0, -30.0, 0.0])},
    4: {"name": "Neutral",   "pos": np.array([-30.0, -30.0, 0.0])},
}

# Corner marker definitions in Paper Frame {M} (centers at +-75 mm, +-75 mm)
MARKER_SIZE_MM = 28.0
MARKER_POSITIONS_MM = {
    0: np.array([-75.0,  75.0, 0.0]),  # Top-Left
    1: np.array([ 75.0,  75.0, 0.0]),  # Top-Right
    2: np.array([ 75.0, -75.0, 0.0]),  # Bottom-Right
    3: np.array([-75.0, -75.0, 0.0]),  # Bottom-Left
}


def generate_aruco_sheet_image(output_path: str = "project/assets/aruco_target_sheet_A4.png"):
    import cv2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # A4 Dimensions at 300 DPI: 210 mm x 297 mm -> 2480 x 3508 pixels
    # 1 mm = 300 / 25.4 ≈ 11.811 pixels
    scale = 300.0 / 25.4
    width_px = int(210.0 * scale)
    height_px = int(297.0 * scale)

    # Create white canvas
    sheet = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    center_x = width_px // 2
    center_y = height_px // 2

    # Draw Header & Instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, "EE430 Course Project - myCobot 280 Jetson Nano", (center_x - 700, 200), font, 1.3, (30, 30, 30), 3)
    cv2.putText(sheet, "Vision-Guided Affective Target Sheet (A4 - 300 DPI)", (center_x - 550, 260), font, 1.0, (80, 80, 80), 2)
    cv2.putText(sheet, "Place on table within X in [145, 185] mm, Y in [-40, 40] mm", (center_x - 620, 310), font, 0.8, (120, 120, 120), 2)

    # Draw 120mm x 120mm Active Grid
    grid_size_px = int(120.0 * scale)
    half_grid = grid_size_px // 2
    x1, y1 = center_x - half_grid, center_y - half_grid
    x2, y2 = center_x + half_grid, center_y + half_grid

    cv2.rectangle(sheet, (x1, y1), (x2, y2), (0, 0, 0), 4)
    # Crosshair dividers (dividing into 4 quadrants)
    cv2.line(sheet, (center_x, y1), (center_x, y2), (0, 0, 0), 3)
    cv2.line(sheet, (x1, center_y), (x2, center_y), (0, 0, 0), 3)

    # Draw Quadrant Labels
    labels = [
        ("QUADRANT 1: HAPPY", "(-30, +30 mm)", x1 + 40, y1 + 100),
        ("QUADRANT 2: ANGRY", "(+30, +30 mm)", center_x + 40, y1 + 100),
        ("QUADRANT 3: SURPRISE", "(+30, -30 mm)", center_x + 40, center_y + 100),
        ("QUADRANT 4: NEUTRAL", "(-30, -30 mm)", x1 + 40, center_y + 100),
    ]
    for title, coord_str, lx, ly in labels:
        cv2.putText(sheet, title, (lx, ly), font, 0.9, (0, 0, 180), 2)
        cv2.putText(sheet, coord_str, (lx, ly + 50), font, 0.7, (100, 100, 100), 2)

    # Draw ArUco Markers
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    marker_px = int(MARKER_SIZE_MM * scale)

    for marker_id, pos_mm in MARKER_POSITIONS_MM.items():
        # Marker center in image pixels
        # Note: In Paper Frame {M}, +X is right, +Y is up. In image, +Y is down.
        mc_x = int(center_x + pos_mm[0] * scale)
        mc_y = int(center_y - pos_mm[1] * scale)

        try:
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        except AttributeError:
            marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_px)

        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        top_left_x = mc_x - marker_px // 2
        top_left_y = mc_y - marker_px // 2

        sheet[top_left_y:top_left_y + marker_px, top_left_x:top_left_x + marker_px] = marker_bgr
        cv2.putText(sheet, f"ID {marker_id}", (mc_x - 30, top_left_y - 15), font, 0.7, (0, 0, 0), 2)

    # Draw Paper Coordinate Frame indicator at center (0,0)
    cv2.circle(sheet, (center_x, center_y), 8, (0, 0, 255), -1)
    cv2.arrowedLine(sheet, (center_x, center_y), (center_x + 100, center_y), (0, 0, 255), 3, tipLength=0.2)
    cv2.arrowedLine(sheet, (center_x, center_y), (center_x, center_y - 100), (0, 200, 0), 3, tipLength=0.2)
    cv2.putText(sheet, "+X_M", (center_x + 110, center_y + 10), font, 0.7, (0, 0, 255), 2)
    cv2.putText(sheet, "+Y_M", (center_x - 20, center_y - 110), font, 0.7, (0, 200, 0), 2)

    cv2.imwrite(output_path, sheet)
    print(f"Generated printable A4 target sheet at: {output_path}")


if __name__ == "__main__":
    generate_aruco_sheet_image()

