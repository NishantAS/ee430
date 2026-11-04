"""
EE430 demo: show 1-4 fingers to the robot's camera -> the arm hovers over that quadrant of the target sheet.

    python main.py                         # 1 trial, MediaPipe counter
    python main.py --trials 10 --log       # 10 trials, asks for ground truth after each -> results/trials.csv
    python main.py --classifier classical  # classical skin/convexity baseline instead
    python main.py --mock                  # no robot (motion simulated), camera still used

Needs calibration first: python calibrate.py all   (see README).

One trial:
  1. go to the calibrated overview pose (camera looking straight down at the table)
  2. finger count must be the same (1-4) for GESTURE_STABLE_FRAMES frames  [keys 1-4 in the window override]
  3. remove the hand; the sheet's ArUco markers give sheet->pixel, calibration gives pixel->robot base
  4. pymycobot send_coords (robot's built-in IK, overview orientation = tool pointing down):
     approach above the quadrant, then a straight-line descent to hover height
  5. read back get_coords, log the error, return to the overview pose
"""

import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np

import config
from common import calibration_store as store
from common.camera import Camera, put_lines, show
from common.robot_client import RobotError, connect, tool_xyz
from vision import sheet as sh
from vision.gestures import CLASSIFIERS, make_counter

WINDOW = "EE430 demo"


class Abort(Exception):
    pass


def check_quit(key):
    if key in (ord("q"), 27):
        raise Abort()


def read_gesture(cam, counter):
    """Returns (quadrant 1-4, source, mean latency ms) once the count is stable, or (None, reason, latency)."""
    history, latencies = deque(maxlen=config.GESTURE_STABLE_FRAMES), []
    t0 = time.time()
    while time.time() - t0 < config.GESTURE_TIMEOUT_S:
        frame = cam.read()
        if frame is None:
            continue
        n = counter.count(frame)
        latencies.append(counter.latency_ms)
        history.append(n)
        stable = len(history) == history.maxlen and len(set(history)) == 1 and history[0] in (1, 2, 3, 4)
        vis = counter.draw(frame)
        put_lines(vis, ["Show 1-4 fingers under the camera  (keys 1-4 override, q quit)",
                        f"{counter.name}: {'no hand' if n is None else f'{n} finger(s)'}  "
                        f"[{sum(h == n for h in history)}/{history.maxlen}]  {counter.latency_ms:.1f} ms"])
        key = show(vis, WINDOW)
        check_quit(key)
        if ord("1") <= key <= ord("4"):
            return key - ord("0"), "keyboard", float(np.mean(latencies))
        if stable:
            return history[0], counter.name, float(np.mean(latencies))
    return None, "timeout", float(np.mean(latencies) if latencies else 0.0)


def locate_sheet(cam, detector):
    """Waits until all 4 markers are visible (hand removed) and averages the sheet homography."""
    hs, t0 = [], time.time()
    while time.time() - t0 < config.SHEET_TIMEOUT_S:
        frame = cam.read()
        if frame is None:
            continue
        markers = detector.detect(frame)
        h = detector.sheet_to_pixel(markers)
        if len(markers) == 4:
            hs.append(h / h[2, 2])
        check_quit(show(put_lines(detector.draw(frame, markers, h),
                                  [f"Remove your hand - markers {len(markers)}/4"]), WINDOW))
        if len(hs) >= config.SHEET_AVERAGE_FRAMES:
            return np.mean(hs, axis=0), frame
    return None, None


def run_trial(i, robot, cam, counter, detector, calib, log_truth):
    tab = calib["table"]
    h_px_base, table_z = np.array(tab["H_pixel_to_base"]), tab["table_z_mm"]
    row = {"trial": i, "classifier": counter.name}
    print(f"\n===== TRIAL {i} =====")

    robot.move_angles(calib["overview_angles"])
    quadrant, source, lat = read_gesture(cam, counter)
    row.update(gesture=quadrant, gesture_source=source, gesture_latency_ms=round(lat, 2))
    if quadrant is None:
        print("No stable gesture.")
        return {**row, "status": "NO_GESTURE"}
    print(f"Gesture: {quadrant} finger(s) ({source}, {lat:.1f} ms/frame)")

    h_sheet_px, frame = locate_sheet(cam, detector)
    if h_sheet_px is None:
        print("Sheet not found (all 4 markers must be visible).")
        return {**row, "status": "NO_SHEET"}
    x, y = sh.quadrant_base_xy(h_sheet_px, h_px_base, quadrant)
    hover = np.array([x, y, table_z + config.HOVER_HEIGHT_MM])
    approach = hover + [0, 0, config.APPROACH_EXTRA_MM]
    row.update(target_x=round(x, 2), target_y=round(y, 2), target_z=round(hover[2], 2))
    print(f"Quadrant {quadrant} -> base ({x:.1f}, {y:.1f}) mm, hover z {hover[2]:.1f} mm")
    cv2.imwrite(str(config.RESULTS_DIR / f"trial_{i:02d}_overview.jpg"),
                detector.draw(frame, detector.detect(frame), h_sheet_px, highlight=quadrant))

    orient, staging = store.hover_orientation(calib), store.staging_angles(calib)
    try:
        robot.move_angles(staging)          # elbow-up start, so the robot's IK keeps the inverted-V shape
        robot.move_tool(approach, orient)
        reached = robot.move_tool(hover, orient)
    except RobotError as e:
        print(e)
        robot.move_angles(staging)
        robot.move_angles(calib["overview_angles"])
        return {**row, "status": "UNREACHABLE"}

    err = float(np.linalg.norm(tool_xyz(reached) - hover))
    row.update(reached_coords=np.round(reached, 2).tolist(), meas_q=np.round(robot.get_angles(), 2).tolist(),
               robot_error_mm=round(err, 2), status="OK")
    print(f"Hover reached: get_coords is {err:.2f} mm from the target")
    hover_frame = cam.read()
    if hover_frame is not None:
        cv2.imwrite(str(config.RESULTS_DIR / f"trial_{i:02d}_hover.jpg"), hover_frame)

    if log_truth:
        truth = input(f"   Fingers you actually showed [{quadrant}]: ").strip()
        miss = input("   Measured offset of tool from the quadrant cross, mm (ENTER = skip): ").strip()
        row["true_gesture"] = int(truth) if truth else quadrant
        row["measured_miss_mm"] = float(miss) if miss else ""
    time.sleep(1.0)
    robot.move_tool(approach, orient)
    robot.move_angles(staging)
    robot.move_angles(calib["overview_angles"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--classifier", choices=CLASSIFIERS, default="mediapipe")
    ap.add_argument("--log", action="store_true", help="ask for ground truth after each trial")
    ap.add_argument("--host", default=config.DEFAULT_HOST)
    ap.add_argument("--mock", action="store_true", help="no robot")
    ap.add_argument("--cam", type=int, help="camera index override")
    args = ap.parse_args()

    calib = store.load(require=["overview_angles", "overview_coords", "table"] + ([] if args.cam is not None else ["camera_index"]))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    robot = connect(args.host, args.mock)
    cam = Camera(args.cam if args.cam is not None else calib["camera_index"])
    counter, detector = make_counter(args.classifier), sh.SheetDetector()

    rows = []
    try:
        for i in range(1, args.trials + 1):
            rows.append(run_trial(i, robot, cam, counter, detector, calib, args.log))
    except Abort:
        print("Stopped by user.")
    finally:
        cv2.destroyAllWindows()
        cam.close()
        try:
            robot.move_angles(calib["overview_angles"])
        except RobotError as e:
            print(f"Could not return to overview: {e}")
        robot.close()

    if rows:
        out = config.RESULTS_DIR / ("MOCK_trials.csv" if args.mock else "trials.csv")
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        ok = [r for r in rows if r["status"] == "OK"]
        print(f"\n{len(ok)}/{len(rows)} trials reached a hover pose. Log: {out}")
        if ok:
            print(f"Mean get_coords-to-target error: {np.mean([r['robot_error_mm'] for r in ok]):.2f} mm")
        if args.log and ok:
            correct = sum(r.get("true_gesture") == r["gesture"] for r in ok)
            print(f"Gesture correct: {correct}/{len(ok)}")


if __name__ == "__main__":
    main()
