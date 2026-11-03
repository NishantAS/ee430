"""
Experiments that produce the report's tables (all written to results/).

    python experiments.py fk          # A2: our DH FK vs the robot's get_coords over 15 poses -> fk_error_table.csv
    python experiments.py ik          # A3: robot's built-in IK (send_coords) from two arm configurations,
                                      #     reachable + unreachable targets -> ik_case_study.csv
    python experiments.py collect     # B/E: save labelled gesture images (keys 0-5) to dl/dataset/<label>/
    python experiments.py gestures    # B/E: classical vs MediaPipe accuracy + latency -> gesture_metrics.csv

Options: --host <jetson ip>, --mock, --cam N
"""

import argparse
import csv
import time

import cv2
import numpy as np

import config
from common import calibration_store as store
from common.robot_client import RobotError, connect
from kinematics import dh_model

# 15 joint configurations (deg) spread over the workspace, all high above the table (checked before moving).
FK_POSES = [
    [0, 0, 0, 0, 0, 0], [0, -20, -40, 0, 0, 0], [30, -20, -40, 20, 0, 0], [-30, -20, -40, 20, 0, 0],
    [0, -30, -60, 0, 0, 45], [45, 10, -30, -20, 20, 0], [-45, 10, -30, -20, -20, 0], [0, 20, 30, -40, 0, 90],
    [60, -10, -20, 30, 30, -45], [-60, -10, -20, 30, -30, 45], [90, 0, -45, 0, 45, 0], [-90, 0, -45, 0, -45, 0],
    [15, -35, -30, -20, 10, 30], [-15, -35, -30, -20, -10, -30], [0, 30, -70, 40, 0, 0],
]

# A3: two arm configurations with the flange at the SAME place (about (190, -60, 240), pointing down):
# elbow bent one way or the other. send_coords is issued from each, so we see which IK branch the robot picks.
IK_START_CONFIGS = {"elbow-up start": [1.4, -18.3, -38.7, -32.9, 0.0, 1.4],
                    "elbow-down start": [1.4, -54.2, 38.7, -74.5, 0.0, 1.4]}
# Flange targets (x, y, height above the table in mm); orientation = calibrated overview (pointing down).
IK_CASES = [
    ("sheet centre", (200.0, -60.0, 60.0)),
    ("quadrant 1 side", (170.0, -30.0, 60.0)),
    ("quadrant 3 side", (230.0, -90.0, 60.0)),
    ("FAIL? close to base (r ~ 125 mm)", (120.0, -40.0, 60.0)),
    ("FAIL? beyond reach (r = 300 mm)", (300.0, 0.0, 60.0)),
    ("FAIL? too high for a vertical tool", (160.0, -60.0, 320.0)),
]


def write_csv(name, rows, mock=False):
    """Mock runs are prefixed MOCK_ so simulated numbers can never be mistaken for lab measurements."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / (("MOCK_" if mock else "") + name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


r2 = lambda v: [round(float(x), 2) for x in v]


# --------------------------------------------------------------------------- A2
def cmd_fk(args):
    robot = connect(args.host, args.mock)
    rows = []
    for i, q_cmd in enumerate(FK_POSES, 1):
        if dh_model.fk(q_cmd)[2, 3] < 150:
            print(f"[{i:02d}] skipped: flange would be too low")
            continue
        robot.move_angles(q_cmd)
        q = robot.get_angles()
        fw = np.array(robot.get_coords(), float)
        t = dh_model.fk(q)
        pos_err = float(np.linalg.norm(t[:3, 3] - fw[:3]))
        rot_err = dh_model.rotation_error_deg(t[:3, :3], dh_model.euler_zyx_to_matrix(*fw[3:]))
        rows.append({"pose": i, "cmd_q": r2(q_cmd), "meas_q": r2(q), "dh_fk": r2(dh_model.flange_coords(q)),
                     "get_coords": r2(fw), "pos_error_mm": round(pos_err, 3), "rot_error_deg": round(rot_err, 3)})
        print(f"[{i:02d}] position error {pos_err:6.2f} mm   orientation error {rot_err:6.2f} deg")
    robot.move_angles(config.HOME_ANGLES)
    if not rows:
        return
    pe, re = [r["pos_error_mm"] for r in rows], [r["rot_error_deg"] for r in rows]
    print(f"\nNominal DH: mean/max position error {np.mean(pe):.2f}/{np.max(pe):.2f} mm, "
          f"orientation {np.mean(re):.2f}/{np.max(re):.2f} deg")

    # Link-length identification: fit on the odd-numbered poses, test on the even ones (held out),
    # so the improvement is not just the fit memorising its own data. Then refit on all poses.
    qs = [r["meas_q"] for r in rows]
    ps = [r["get_coords"][:3] for r in rows]
    train = [i for i in range(len(rows)) if i % 2 == 0]
    test = [i for i in range(len(rows)) if i % 2 == 1]
    fitted_train = dh_model.identify_lengths([qs[i] for i in train], [ps[i] for i in train])
    fitted_all = dh_model.identify_lengths(qs, ps)
    err = lambda L, i: float(np.linalg.norm(dh_model.fk(qs[i], L)[:3, 3] - ps[i]))
    for i, r in enumerate(rows):
        r["fitted_pos_error_mm"] = round(err(fitted_train, i), 3)
        r["fit_role"] = "train" if i in train else "test"
    print("Held-out poses: mean position error "
          f"{np.mean([err(dh_model.NOMINAL_LENGTHS, i) for i in test]):.2f} mm (nominal) -> "
          f"{np.mean([err(fitted_train, i) for i in test]):.2f} mm (fitted lengths)")
    print("Link lengths (mm):  " + "  ".join(
        f"{n} {nom:.2f}->{fit:.2f}" for n, nom, fit in zip(dh_model.LENGTH_NAMES, dh_model.NOMINAL_LENGTHS, fitted_all)))
    write_csv("fk_error_table.csv", rows, args.mock)
    write_csv("dh_link_lengths.csv", [{"length": n, "nominal_mm": nom, "fitted_mm": round(float(fit), 3)}
                                      for n, nom, fit in zip(dh_model.LENGTH_NAMES, dh_model.NOMINAL_LENGTHS,
                                                             fitted_all)], args.mock)


# --------------------------------------------------------------------------- A3
def cmd_ik(args):
    """
    For every target and both start configurations: send_coords, then read back where the arm went.
    Same target + different resulting joint angles = the robot's IK picked a different solution branch.
    A target the robot never reaches (RobotError after the timeout) is a failure case.
    """
    calib = store.load(require=["overview_coords", "table"])
    orient, table_z = calib["overview_coords"][3:], calib["table"]["table_z_mm"]
    robot = connect(args.host, args.mock)
    rows = []
    for name, (x, y, h) in IK_CASES:
        target = [x, y, table_z + h, *orient]
        for start_name, q_start in IK_START_CONFIGS.items():
            robot.move_angles(q_start)
            row = {"case": name, "start": start_name, "target_coords": r2(target)}
            t0 = time.time()
            try:
                reached = robot.move_coords(target)
                q = robot.get_angles()
                row.update(reached=True, final_q=r2(q), reached_coords=r2(reached),
                           pos_error_mm=round(float(np.linalg.norm(np.subtract(reached[:3], target[:3]))), 2))
                msg = f"q = {r2(q)}"
            except RobotError as e:
                q = robot.get_angles()
                row.update(reached=False, final_q=r2(q), note=str(e))
                msg = "NOT reached"
            row["time_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"{name:<36} {start_name:<17} {msg}")
            if not row["reached"]:  # get away from wherever it stopped before the next start pose
                robot.move_angles(config.HOME_ANGLES)
    robot.move_angles(config.HOME_ANGLES)
    write_csv("ik_case_study.csv", rows, args.mock)


# --------------------------------------------------------------------------- gestures
def cmd_collect(args):
    """Saves frames from the flange camera, labelled by the key you press (0-5 = number of fingers)."""
    from common.camera import Camera, put_lines, show
    cam = Camera(args.cam if args.cam is not None else store.load(require=["camera_index"])["camera_index"])
    counts = {n: len(list((config.DATASET_DIR / str(n)).glob("*.jpg"))) for n in range(6)}
    print("Put the arm in the overview pose first (python robot_tools.py overview).")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            vis = put_lines(frame.copy(), ["Show a gesture, press 0-5 = fingers shown, q = quit",
                                           "saved: " + "  ".join(f"{n}:{c}" for n, c in counts.items())])
            key = show(vis, "EE430 collect")
            if key in (ord("q"), 27):
                break
            if ord("0") <= key <= ord("5"):
                n = key - ord("0")
                folder = config.DATASET_DIR / str(n)
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{counts[n]:04d}.jpg"), frame)
                counts[n] += 1
    finally:
        cam.close()
        cv2.destroyAllWindows()


def cmd_gestures(args):
    """Both counters are pretrained / rule-based, so the whole lab dataset is a held-out test set."""
    from vision.gestures import CLASSIFIERS, make_counter
    images = sorted((int(p.parent.name), p) for p in config.DATASET_DIR.glob("[0-5]/*.jpg"))
    if not images:
        raise SystemExit(f"No images in {config.DATASET_DIR}. Run: python experiments.py collect")
    print(f"{len(images)} labelled images")
    summary, predictions = [], []
    for name in CLASSIFIERS:
        counter = make_counter(name)
        conf = np.zeros((6, 7), int)  # rows: true 0-5, cols: predicted 0-5 or "no hand"
        lat = []
        for label, path in images:
            pred = counter.count(cv2.imread(str(path)))
            lat.append(counter.latency_ms)
            conf[label, 6 if pred is None else min(pred, 5)] += 1
            predictions.append({"classifier": name, "image": path.name, "true": label, "pred": pred})
        acc = np.trace(conf[:, :6]) / conf.sum()
        task = conf[1:5]  # the 4 gestures the demo uses
        task_acc = np.trace(task[:, 1:5]) / max(task.sum(), 1)
        summary.append({"classifier": name, "images": int(conf.sum()), "accuracy_all": round(float(acc), 3),
                        "accuracy_1to4": round(float(task_acc), 3),
                        "no_hand_rate": round(float(conf[:, 6].sum() / conf.sum()), 3),
                        "latency_ms_mean": round(float(np.mean(lat)), 2),
                        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
                        "device": "laptop CPU"})
        print(f"\n{name}: accuracy {acc:.1%} (1-4 fingers: {task_acc:.1%}), {np.mean(lat):.1f} ms/frame")
        print("confusion (rows true 0-5, cols pred 0-5, none):\n", conf)
    write_csv("gesture_metrics.csv", summary)
    write_csv("gesture_predictions.csv", predictions)


if __name__ == "__main__":
    cmds = {"fk": cmd_fk, "ik": cmd_ik, "collect": cmd_collect, "gestures": cmd_gestures}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=cmds)
    ap.add_argument("--host", default=config.DEFAULT_HOST)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--cam", type=int)
    a = ap.parse_args()
    cmds[a.experiment](a)
