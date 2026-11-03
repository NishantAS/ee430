"""
Target sheet: geometry, printable generator, detection and pixel <-> table mapping.

Sheet frame {M} (mm): origin at the sheet centre, +X right, +Y up (as printed), Z out of the paper.
Four ArUco markers (DICT_4X4_50, IDs 0-3, 28 mm) at (+-75, +-75); a 120 mm grid split into 4 quadrants.

Mapping chain used by the demo (all homographies of the table plane):
    sheet mm --H_sheet_to_pixel (from the detected marker corners, every trial)--> image pixel
    image pixel --H_pixel_to_base (calibrate.py table, fixed for the overview pose)--> robot base (x, y)
so the sheet may be moved between trials without recalibrating.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config

MARKER_SIZE_MM = 28.0
MARKER_CENTERS_MM = {0: (-75.0, 75.0), 1: (75.0, 75.0), 2: (75.0, -75.0), 3: (-75.0, -75.0)}
QUADRANT_CENTERS_MM = {1: (-30.0, 30.0), 2: (30.0, 30.0), 3: (30.0, -30.0), 4: (-30.0, -30.0)}
SHEET_CENTER_MM = (0.0, 0.0)

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _marker_corners_mm(marker_id: int) -> np.ndarray:
    """Corner order as returned by OpenCV: top-left, top-right, bottom-right, bottom-left."""
    cx, cy = MARKER_CENTERS_MM[marker_id]
    s = MARKER_SIZE_MM / 2.0
    return np.array([[cx - s, cy + s], [cx + s, cy + s], [cx + s, cy - s], [cx - s, cy - s]], np.float32)


def apply_h(h: np.ndarray, pts) -> np.ndarray:
    """Applies a 3x3 homography to an (N, 2) or (2,) array of points."""
    pts = np.asarray(pts, np.float64)
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    return out[0] if pts.ndim == 1 else out


class SheetDetector:
    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(DICTIONARY, params)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # evens out glare / dim spots

    def detect(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """{marker id: 4x2 corner pixels} for the sheet's markers found in the frame."""
        gray = self._clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners) if int(i) in MARKER_CENTERS_MM}

    @staticmethod
    def sheet_to_pixel(markers: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """Homography sheet mm -> pixel from all detected marker corners (needs >= 2 markers = 8 points)."""
        if len(markers) < 2:
            return None
        src = np.vstack([_marker_corners_mm(i) for i in markers])
        dst = np.vstack(list(markers.values())).astype(np.float32)
        h, _ = cv2.findHomography(src, dst)
        return h

    @staticmethod
    def marker_centers_px(markers: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return {i: c.mean(axis=0) for i, c in markers.items()}

    @staticmethod
    def draw(frame: np.ndarray, markers: Dict[int, np.ndarray], h_sheet_px: Optional[np.ndarray] = None,
             highlight: Optional[int] = None) -> np.ndarray:
        vis = frame.copy()
        if markers:
            cv2.aruco.drawDetectedMarkers(vis, [c.reshape(1, 4, 2) for c in markers.values()],
                                          np.array(list(markers.keys())))
        if h_sheet_px is not None:
            grid = apply_h(h_sheet_px, [(-60, 60), (60, 60), (60, -60), (-60, -60)]).astype(np.int32)
            cv2.polylines(vis, [grid], True, (255, 200, 0), 2)
            for q, xy in QUADRANT_CENTERS_MM.items():
                p = tuple(int(v) for v in apply_h(h_sheet_px, xy))
                color = (0, 0, 255) if q == highlight else (0, 200, 0)
                cv2.circle(vis, p, 9 if q == highlight else 5, color, -1)
                cv2.putText(vis, f"Q{q}", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis


def quadrant_base_xy(h_sheet_px: np.ndarray, h_px_base: np.ndarray, quadrant: int) -> Tuple[float, float]:
    """Quadrant centre in the robot base frame (mm): sheet mm -> pixel -> base."""
    px = apply_h(h_sheet_px, QUADRANT_CENTERS_MM[quadrant])
    x, y = apply_h(h_px_base, px)
    return float(x), float(y)


def generate_printable(path=config.SHEET_IMAGE, dpi: int = 300) -> None:
    """Writes the A4 target sheet. Print at 100 % scale (no 'fit to page') and check a marker is 28 mm."""
    mm = dpi / 25.4
    w, h = int(210 * mm), int(297 * mm)
    sheet = np.full((h, w), 255, np.uint8)
    cx, cy = w // 2, h // 2

    def to_px(x, y):
        return int(round(cx + x * mm)), int(round(cy - y * mm))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, "EE430 myCobot 280 - gesture target sheet", (int(20 * mm), int(20 * mm)), font, 2.0, 0, 4)
    cv2.putText(sheet, "Print at 100% scale. Each marker must measure 28 mm.", (int(20 * mm), int(30 * mm)),
                font, 1.3, 90, 3)

    cv2.rectangle(sheet, to_px(-60, 60), to_px(60, -60), 0, 6)
    cv2.line(sheet, to_px(0, 60), to_px(0, -60), 0, 4)
    cv2.line(sheet, to_px(-60, 0), to_px(60, 0), 0, 4)
    for q, (x, y) in QUADRANT_CENTERS_MM.items():
        cv2.drawMarker(sheet, to_px(x, y), 0, cv2.MARKER_CROSS, int(8 * mm), 4)
        px, py = to_px(x - 25, y + 20)
        cv2.putText(sheet, f"{q} finger{'s' if q > 1 else ''}", (px, py), font, 2.0, 60, 4)

    size = int(round(MARKER_SIZE_MM * mm))
    for i, (x, y) in MARKER_CENTERS_MM.items():
        img = cv2.aruco.generateImageMarker(DICTIONARY, i, size)
        x0, y0 = to_px(x, y)
        x0, y0 = x0 - size // 2, y0 - size // 2
        sheet[y0:y0 + size, x0:x0 + size] = img
        cv2.putText(sheet, f"ID {i}", (x0, y0 - 20), font, 1.5, 0, 3)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    print(f"Printable sheet written to {path}")
