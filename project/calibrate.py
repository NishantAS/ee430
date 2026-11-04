"""
Calibration: run these BEFORE main.py. Results go to calibration/calibration.json.

    python calibrate.py sheet       # write the printable A4 target sheet (print at 100 %)
    python calibrate.py camera      # pick which laptop camera index is the flange camera
    python calibrate.py overview    # move the arm BY HAND above the sheet; it straightens the camera, saves
    python calibrate.py table       # touch the 4 markers + sheet centre BY HAND -> pixel->robot homography
    python calibrate.py check       # visit all 4 quadrants with the calibration to verify it
    python calibrate.py all         # camera, overview, table, check in sequence

Camera-to-robot calibration (criterion B2): from the fixed overview joint pose the camera sees the table
plane, so image pixels and table points (x, y) in the robot base frame are related by a homography H.
H is fitted from 4 marker centres seen in the image and touched with the tool tip (positions read from the
robot's get_coords), and checked on a 5th touched point (sheet centre) that is not used in the fit.

Common options: --host <jetson ip>, --mock (no robot), --cam <index> (override the stored camera).
"""

import argparse

import cv2
import numpy as np

import config
from common import calibration_store as store
from common.camera import Camera, put_lines, show
from common.teach import teach
from common.robot_client import RobotError, connect, elbow_clearly_up
from kinematics import dh_model
from vision import sheet as sh


class Session:
    """
    One robot connection and one open camera, shared by every step of a run (the camera device can only be
    opened once at a time, so `calibrate.py all` must not reopen it per step). Opened on first use.
    """

    def __init__(self, args):
        self.args, self._robot, self._cam = args, None, None

    @property
    def robot(self):
        if self._robot is None:
            self._robot = connect(self.args.host, self.args.mock)
        return self._robot

    @property
    def cam(self) -> Camera:
        if self._cam is None:
            index = self.args.cam if self.args.cam is not None else store.load().get("camera_index")
            if index is None:
                raise SystemExit("No camera selected. Run: python calibrate.py camera")
            self._cam = Camera(index)
        return self._cam

    def close(self):
        if self._cam is not None:
            self._cam.close()
        if self._robot is not None:
            self._robot.close()


def detect_sheet(cam: Camera, detector: sh.SheetDetector, frames: int = 15, need_all: bool = True):
    """Averages marker centres over several frames. Returns ({id: centre px}, last frame, H_sheet_px)."""
    acc = {i: [] for i in sh.MARKER_CENTERS_MM}
    got, frame, markers = 0, None, {}
    for _ in range(frames * 10):
        frame = cam.read()
        if frame is None:
            continue
        markers = detector.detect(frame)
        show(detector.draw(frame, markers, detector.sheet_to_pixel(markers)), "EE430 calibration")
        if need_all and len(markers) < 4:
            continue
        for i, c in detector.marker_centers_px(markers).items():
            acc[i].append(c)
        got += 1
        if got >= frames:
            break
    centres = {i: np.mean(v, axis=0) for i, v in acc.items() if v}
    return centres, frame, detector.sheet_to_pixel(markers)


# --------------------------------------------------------------------------- steps
def cmd_sheet(s: Session):
    sh.generate_printable()


def cmd_camera(s: Session):
    """Shows each camera index; press y to use it, any other key for the next one."""
    for idx in range(8):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        print(f"camera {idx}: y = use this one, any other key = next")
        key = -1
        while key == -1 or key == 255:
            ok, frame = cap.read()
            if not ok:
                break
            key = show(put_lines(frame, [f"camera index {idx}: y = use, other key = next"]), "EE430 camera", 30)
        cap.release()
        if key == ord("y"):
            cv2.destroyAllWindows()
            store.update(camera_index=idx)
            return
    cv2.destroyAllWindows()
    raise SystemExit("No camera chosen. Is the flange camera plugged into the laptop?")


def go_overview(robot, calib):
    """Overview pose by joint angles (send_angles): exactly repeatable, which the homography relies on."""
    robot.move_angles(calib["overview_angles"])


MAX_TOUCH_TILT_DEG = 15.0   # touches held further from vertical are retaken


def tool_tip(coords, angles) -> np.ndarray:
    """
    Tool tip in the base frame for a hand-taught (possibly slightly tilted) pose: flange position from the
    robot's get_coords + TOOL_LENGTH_MM along the flange z-axis. The axis direction comes from our DH FK of the
    joint angles (its orientation matches the robot's, see robot_tools.py status).
    """
    return np.array(coords[:3], float) + config.TOOL_LENGTH_MM * dh_model.fk(angles)[:3, 2]


def wait_key(cam, title, lines, overlay=None) -> int:
    """Shows the live view until ENTER, SPACE or ESC is pressed; returns that key."""
    while True:
        frame = cam.read()
        if frame is None:
            continue
        extra = overlay(frame) if overlay else []
        key = show(put_lines(frame, [title, *lines, *extra]), "EE430 teach")
        if key in (13, 32, 27):
            return key


def cmd_overview(s: Session):
    calib = store.load(require=["camera_index"] if s.args.cam is None else [])
    robot, cam = s.robot, s.cam
    detector = sh.SheetDetector()

    def overlay(frame):
        markers = detector.detect(frame)
        frame[:] = detector.draw(frame, markers, detector.sheet_to_pixel(markers))
        return [f"markers visible: {len(markers)}/4"]

    # start from an elbow-up pose (by joint angles, no IK involved) so there is less to move by hand
    robot.move_angles(calib.get("overview_angles", config.DEFAULT_OVERVIEW_ANGLES))
    print("Move the arm by hand so the camera looks down at the sheet with all 4 markers green, then ENTER.\n"
          "Keep the elbow bent UP (inverted V) and the camera no higher than ~250 mm above the robot base.")
    while True:
        taught = teach(robot, cam, "OVERVIEW: camera above the sheet, all 4 markers green", overlay)
        if taught is None:
            raise SystemExit("Aborted, nothing saved.")
        coords, angles = taught
        if not elbow_clearly_up(angles):
            print(f"   Elbow is not bent up enough (q3 = {angles[2]:.0f} deg, need <= "
                  f"{-config.ELBOW_UP_MIN_BEND_DEG:.0f}): later moves could fold the arm into a V and hit the "
                  "base. Teach again with the elbow higher (lower the camera a little if needed).")
            continue
        # Hand placement is never exactly vertical: let the robot straighten the camera to point straight down
        # (rx = 180, ry = 0), keeping the position and the rotation about the vertical axis (rz).
        try:
            robot.move_coords([*coords[:3], 180.0, 0.0, coords[5]], config.SLOW_SPEED)
        except RobotError as e:
            print(f"   Could not straighten the camera here ({e}). Try a slightly lower pose (z <= ~250 mm).")
            continue
        key = wait_key(cam, "OVERVIEW (camera straightened)",
                       ["ENTER = save this pose   SPACE = teach again   ESC = abort"], overlay)
        if key == 13:
            break
        if key == 27:
            raise SystemExit("Aborted, nothing saved.")
    # angles to come back to exactly this pose; coords (the robot's own reading) give the hover orientation
    store.update(overview_angles=[round(a, 2) for a in robot.get_angles()],
                 overview_coords=[round(c, 2) for c in robot.get_coords()])


def cmd_table(s: Session):
    calib = store.load(require=["overview_angles", "overview_coords"]
                       + ([] if s.args.cam is not None else ["camera_index"]))
    robot, cam = s.robot, s.cam
    detector = sh.SheetDetector()

    go_overview(robot, calib)
    print("Detecting the sheet from the overview pose ... (do NOT move the sheet until this step is finished)")
    centres, _, _ = detect_sheet(cam, detector)
    if len(centres) < 4:
        raise SystemExit("Not all 4 markers visible from the overview pose. Re-run: python calibrate.py overview")
    pixel_pts = [centres[i] for i in sorted(sh.MARKER_CENTERS_MM)]

    targets = [f"centre of marker ID {i}" for i in sorted(sh.MARKER_CENTERS_MM)] + ["sheet centre (check)"]
    robot_pts = []
    print(f"For each point: hold the arm, SPACE to release, put the tool tip ({config.TOOL_LENGTH_MM:g} mm below "
          "the flange) on the point, keep the tool roughly vertical, ENTER to lock and record.")
    for label in targets:
        while True:
            taught = teach(robot, cam, f"TOUCH: {label}")
            if taught is None:
                raise SystemExit("Aborted, nothing saved.")
            coords, angles = taught
            tilt = float(np.degrees(np.arccos(np.clip(-dh_model.fk(angles)[2, 2], -1.0, 1.0))))
            if tilt <= MAX_TOUCH_TILT_DEG:
                break
            print(f"   tool is tilted {tilt:.0f} deg from vertical (max {MAX_TOUCH_TILT_DEG:g}) - touch it again, "
                  "holding the tool more upright")
        p = tool_tip(coords, angles)
        robot_pts.append(p)
        print(f"   {label:<24} -> base ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
        try:  # lift straight up so the tip does not drag over the sheet
            robot.move_coords([coords[0], coords[1], coords[2] + 30.0, *coords[3:]], config.SLOW_SPEED)
        except RobotError as e:
            print(f"   (could not lift: {e})")

    robot_pts = np.array(robot_pts)
    h, _ = cv2.findHomography(np.array(pixel_pts), robot_pts[:4, :2])
    table_z = float(robot_pts[:4, 2].mean())
    # sheet centre in the image: map (0, 0) through the sheet->pixel homography of the 4 marker centres
    h_sheet_px, _ = cv2.findHomography(np.array([sh.MARKER_CENTERS_MM[i] for i in sorted(centres)]),
                                       np.array(pixel_pts))
    predicted = sh.apply_h(h, sh.apply_h(h_sheet_px, sh.SHEET_CENTER_MM))
    check_err = float(np.linalg.norm(predicted - robot_pts[4, :2]))
    z_spread = float(np.ptp(robot_pts[:, 2]))
    print(f"\nTable height z = {table_z:.1f} mm (spread {z_spread:.1f} mm over 5 touches; > 5 mm means "
          "the table/sheet is not flat or a touch was off)")
    print(f"Check point error (sheet centre, not used in fit): {check_err:.1f} mm")

    go_overview(robot, calib)
    store.update(table={
        "pixel_points": [[round(float(v), 2) for v in p] for p in pixel_pts],
        "robot_points": [[round(float(v), 2) for v in p] for p in robot_pts],
        "H_pixel_to_base": h.tolist(),
        "table_z_mm": round(table_z, 2),
        "check_error_mm": round(check_err, 2),
    })


def cmd_check(s: Session):
    """Moves over each quadrant cross using the stored calibration; you judge / measure the offset."""
    calib = store.load(require=["overview_angles", "overview_coords", "table"]
                       + ([] if s.args.cam is not None else ["camera_index"]))
    robot, cam = s.robot, s.cam
    detector = sh.SheetDetector()
    h_px_base = np.array(calib["table"]["H_pixel_to_base"])
    table_z, orient = calib["table"]["table_z_mm"], store.hover_orientation(calib)

    go_overview(robot, calib)
    _, _, h_sheet_px = detect_sheet(cam, detector, frames=3, need_all=False)
    if h_sheet_px is None:
        raise SystemExit("Sheet not visible from the overview pose.")
    robot.move_angles(store.staging_angles(calib))   # elbow-up start for the robot's IK
    for q in sh.QUADRANT_CENTERS_MM:
        x, y = sh.quadrant_base_xy(h_sheet_px, h_px_base, q)
        above, hover = table_z + config.HOVER_HEIGHT_MM + config.APPROACH_EXTRA_MM, table_z + config.HOVER_HEIGHT_MM
        print(f"Quadrant {q}: base ({x:.1f}, {y:.1f}) mm")
        robot.move_tool((x, y, above), orient)
        robot.move_tool((x, y, hover), orient)
        input("   Tool should be straight above the cross. Press ENTER for the next quadrant ...")
        robot.move_tool((x, y, above), orient)
    robot.move_angles(store.staging_angles(calib))
    go_overview(robot, calib)
    print("Check finished.")


def cmd_all(s: Session):
    for step in (cmd_camera, cmd_overview, cmd_table, cmd_check):
        print(f"\n===== {step.__name__[4:].upper()} =====")
        step(s)


if __name__ == "__main__":
    steps = {"sheet": cmd_sheet, "camera": cmd_camera, "overview": cmd_overview,
             "table": cmd_table, "check": cmd_check, "all": cmd_all}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=steps)
    ap.add_argument("--host", default=config.DEFAULT_HOST)
    ap.add_argument("--mock", action="store_true", help="no robot (motion is simulated)")
    ap.add_argument("--cam", type=int, help="camera index (default: the one saved by 'camera')")
    session = Session(ap.parse_args())
    try:
        steps[session.args.step](session)
    finally:
        cv2.destroyAllWindows()
        session.close()
