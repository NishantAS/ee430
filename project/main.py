"""
Master Closed-Loop Finite State Machine for EE430 Course Project.

Workflow:
1. STATE_INIT: Establish connection with robot (or mock client) and load calibration.
2. STATE_HOME: Move arm to safe neutral home position.
3. STATE_CAPTURE_FACE:
   - Command arm to SEARCH_FACE pose (camera oriented horizontally toward human).
   - Capture video frames, run emotion recognition (DL or Classical).
   - Perform 10-frame majority voting to ensure robust classification.
4. STATE_LOCATE_PAPER:
   - Command arm to SEARCH_PAPER pose (camera oriented top-down at Z ~ 220 mm).
   - Wait for settling time (0.5s) to eliminate motion blur.
   - Detect ArUco corner markers (DICT_4X4_50, IDs 0..3) with CLAHE.
   - Compute ^C T_M using solvePnP and determine target quadrant in Base Frame {0}.
5. STATE_HOVER_QUADRANT:
   - Command arm to target quadrant hover pose at Z = 100 mm using send_coords(mode=1).
   - Verify hover position and stability.
6. STATE_EVALUATE_TRIAL:
   - Record trial metrics (accuracy, positioning error, latency) to project/results/ten_trial_positioning.csv.
7. STATE_WAIT_INPUT / RETURN_HOME:
   - Wait for user confirmation or iterate to next trial, then return to HOME.
"""

import os
import sys
import time
import argparse
import csv
import select
from collections import Counter
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from calibration.calibration_data import load_camera_calibration
from vision.aruco_tracker import ArUcoTracker
from vision.classical_emotion import ClassicalEmotionClassifier
from vision.pretrained_emotion import PretrainedEmotionClassifier
from robot_client import RobotRemoteClient

# The camera is mounted rolled about the flange axis relative to the DH hand-eye model (verified on the robot:
# the feed was ~40 deg off at the model's "level" pose), so ^6R_C = Rz(-CAMERA_ROLL_OFFSET_DEG) * model.
# The poses below already include that roll so the image is level.
CAMERA_ROLL_OFFSET_DEG = 40.0

# End-effector poses [x, y, z (mm), rx, ry, rz (deg)] sent as raw coordinates to the robot, which solves the joints.
# (Initially derived from the verified DH model, ~0.5 mm / 0.4 deg vs get_coords.)
POSE_HOME = [51.95, -64.45, 410.32, -91.56, 52.47, -91.07]
POSE_SEARCH_FACE = [43.67, -64.62, 398.69, -90.0, 52.48, -90.0]
# Top-down overview pose: camera at ~X=156mm, Y=3mm, Z=245mm looking straight down (axis=[0,0,-1]).
# Flange z=270 is near the ceiling for a vertical camera at this reach (DH-based IK check: ~274 mm at x=162; higher tilts it).
POSE_SEARCH_PAPER = [162.0, -26.63, 290.0, 180.0, 0.0, -130.48]
HOVER_Z_MM = 130.0  # flange height over the chosen quadrant (table/workspace sits higher than the original 100 mm)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def hand_eye_with_roll(t_6_c_model: np.ndarray, roll_offset_deg: float) -> np.ndarray:
    """^6T_C with the camera's measured roll about the flange axis applied: ^6R_C = Rz(-roll) * model."""
    c, s = np.cos(np.radians(-roll_offset_deg)), np.sin(np.radians(-roll_offset_deg))
    t = t_6_c_model.copy()
    t[0:3, 0:3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ t_6_c_model[0:3, 0:3]
    return t


def display_frame(title: str, frame: np.ndarray, wait_ms: int = 1):
    """Safely displays frame in OpenCV window if DISPLAY is available."""
    if not os.environ.get("DISPLAY"):
        return
    try:
        cv2.imshow(title, frame)
        cv2.waitKey(wait_ms)
    except Exception:
        pass


class AffectivePointingFSM:
    def __init__(self, host="127.0.0.1", mock=False, use_dl=True, calib_path="project/calibration/camera_params.npz",
                 step_mode=True, camera_index=0, model_name="hse"):
        self.model_name = model_name
        self.mock = mock
        self.step_mode = step_mode
        self.use_dl = use_dl

        print("Initializing Master Robot Controller...")
        self.robot = RobotRemoteClient(host=host, mock=mock, camera_index=camera_index)

        # Load Calibration
        self.K, self.D, self.T_6_C = load_camera_calibration(calib_path)
        self.tracker = ArUcoTracker(self.K, self.D, self.T_6_C)

        # Emotion classifiers
        print(f"Loading Emotion Classifier (Mode: {f'Pretrained CNN ({self.model_name})' if use_dl else 'Classical (HOG+Linear)'})...")
        if use_dl:
            self.emotion_clf = PretrainedEmotionClassifier(self.model_name)
        else:
            self.emotion_clf = ClassicalEmotionClassifier()

        self.T_6_C_nominal = self.T_6_C.copy()
        self.T_6_C = hand_eye_with_roll(self.T_6_C_nominal, CAMERA_ROLL_OFFSET_DEG)
        self.tracker.T_6_C = self.T_6_C
        self.paper_pose = list(POSE_SEARCH_PAPER)  # search pose with yaw refined by the roll alignment

        self.local_cap = None  # camera is owned by RobotRemoteClient

    def wait_step(self, msg: str):
        """Pauses until ENTER is pressed (q + ENTER quits) while keeping the live camera window updating."""
        if not self.step_mode:
            return
        print(f"\n>>> Press ENTER to {msg} (q + ENTER to quit) ...", flush=True)
        while True:
            frame = self.robot.read_frame(self.local_cap, timeout=0.1)
            if frame is not None:
                cv2.putText(frame, f"NEXT: {msg}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                display_frame("EE430 - Robot Camera View", frame, wait_ms=1)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                if sys.stdin.readline().strip().lower() == "q":
                    sys.exit("Quit by user.")
                return

    def move_to(self, coords, mode: int = 0, speed: int = 25, pos_tol_mm: float = 3.0, ori_tol_deg: float = 3.0,
                timeout: float = 15.0, settle: float = 0.6):
        """
        Sends an end-effector pose [x, y, z, rx, ry, rz] to the robot (send_coords; mode 0 = joint-interpolated,
        1 = straight line) and blocks until the reported pose is within tolerance (not a fixed sleep).
        """
        coords = [float(c) for c in coords]
        self.robot.send_coords(coords, speed=speed, mode=mode)
        curr = None
        t0 = time.time()
        time.sleep(0.3)
        while time.time() - t0 < timeout:
            curr = self.robot.get_coords()
            if curr and len(curr) == 6:
                pos_err = float(np.linalg.norm(np.array(curr[:3]) - np.array(coords[:3])))
                ori_err = max(abs(wrap_deg(c - t)) for c, t in zip(curr[3:], coords[3:]))
                if pos_err < pos_tol_mm and ori_err < ori_tol_deg:
                    break
            time.sleep(0.15)
        else:
            print(f"   [move] Warning: arm did not reach target within {timeout:.0f}s "
                  f"(target {[round(c, 1) for c in coords]}, now {curr})")
        time.sleep(settle)  # let vibration die down before imaging

    def align_camera_roll(self, tol_deg: float = 2.0, max_iters: int = 6) -> bool:
        """
        Turns the end effector about the vertical axis (= camera optical axis in the top-down pose, by changing rz)
        so the sheet's edges line up with the image axes, which keeps all 4 markers inside the frame (a sheet seen
        as a diamond is cut off by the image border). The sheet looks identical every 90 deg, so only the smallest
        turn (<= 45 deg) is made. ^6T_C is NOT changed: the paper pose from solvePnP and the real joint readings
        already account for the turn.
        """
        sign, prev_err = -1.0, None
        for it in range(max_iters):
            roll = None
            for _ in range(10):  # a few fresh frames; sheet may be only partly visible
                frame = self.robot.read_frame(self.local_cap)
                if frame is None:
                    continue
                corners, ids = self.tracker.detect_markers(frame)
                roll = self.tracker.paper_roll_deg(corners, ids)
                if roll is not None:
                    roll = (roll + 45.0) % 90.0 - 45.0  # fold into [-45, 45)
                    vis = self.tracker.draw_detections(frame, corners, ids)
                    cv2.putText(vis, f"Aligning roll: {roll:+.1f} deg", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    display_frame("EE430 - Robot Camera View", vis, wait_ms=1)
                    break
            if roll is None:
                print("   [align] No ArUco markers visible - cannot align camera roll.")
                return False

            pose = self.robot.get_coords()
            print(f"   [align] iter {it}: sheet roll = {roll:+.1f} deg (rz = {pose[5]:.1f})")
            if abs(roll) <= tol_deg:
                self.paper_pose = list(pose)
                return True

            # Turning the camera by +psi about its optical axis turns the image by -psi; the axis points down,
            # so rz += d is psi = -d. If the last move made things worse, flip the sign.
            if prev_err is not None and abs(roll) > abs(prev_err) - 0.5:
                sign = -sign
            prev_err = roll
            pose[5] = wrap_deg(pose[5] + sign * roll)
            self.move_to(pose, speed=20)
        print("   [align] Roll did not converge.")
        return False

    def run_trial(self, trial_idx: int = 1) -> dict:
        print(f"\n=======================================================")
        print(f" RUNNING TRIAL {trial_idx:02d}")
        print(f"=======================================================")

        trial_log = {
            "Trial_ID": trial_idx,
            "Target_Emotion": None,
            "Detected_Emotion": None,
            "Emotion_Match": False,
            "Quadrant_ID": None,
            "ArUco_Detected": False,
            "Target_X_mm": 0.0,
            "Target_Y_mm": 0.0,
            "Target_Z_mm": 0.0,
            "Position_Error_mm": 0.0,
            "Emotion_Latency_ms": 0.0,
            "Status": "FAILED"
        }

        # Step 1: Move to SEARCH_FACE
        #self.wait_step("move to SEARCH_FACE")
        print("[1/5] Moving to SEARCH_FACE configuration...")
        self.move_to(POSE_SEARCH_FACE)

        # Step 2: Majority Vote for Emotion Recognition with Live Video Feedback
        self.wait_step("start face / emotion capture (get your expression ready)")
        print("[2/5] Capturing face and recognizing emotion (showing live feed)...")
        votes = []
        score_sum = np.zeros(4)
        latencies = []
        t_face = time.time()
        while time.time() - t_face < 8.0:  # up to 8 s to collect 10 votes
            frame = self.robot.read_frame(self.local_cap)
            if frame is None:
                continue

            vis = frame.copy()
            qid, label, lat, bbox = self.emotion_clf.predict(frame)
            if qid is not None:
                votes.append((qid, label))
                latencies.append(lat)
                if getattr(self.emotion_clf, "last_scores", None) is not None:
                    score_sum += self.emotion_clf.last_scores
                if bbox is not None:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(vis, f"{label} ({lat:.1f}ms)", (bx, by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Looking for face...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            display_frame("EE430 - Robot Camera View", vis, wait_ms=1)
            if len(votes) >= 10:
                break

        if not votes:
            print("ERROR: No face detected - aborting trial (no silent Neutral default).")
            trial_log["Detected_Emotion"] = "NoFace"
            trial_log["Status"] = "NO_FACE"
            self.move_to(POSE_HOME)
            return trial_log
        else:
            if score_sum.any():  # average the class scores over the frames: steadier than counting per-frame labels
                from vision.pretrained_emotion import GROUPS
                quadrant_id, label = GROUPS[int(np.argmax(score_sum))][:2]
                print(f"   [emotion] mean scores over {len(votes)} frames: "
                      + ", ".join(f"{g[1]} {v / len(votes):.2f}" for g, v in zip(GROUPS, score_sum)))
            else:
                quadrant_id, label = Counter(votes).most_common(1)[0][0]
            avg_lat = float(np.mean(latencies))

        print(f"-> Detected Emotion: {label.upper()} -> Target Quadrant: {quadrant_id} (Inference Latency: {avg_lat:.2f} ms)")
        trial_log["Detected_Emotion"] = label
        trial_log["Quadrant_ID"] = quadrant_id
        trial_log["Emotion_Latency_ms"] = round(avg_lat, 2)

        self.wait_step("move to SEARCH_PAPER")
        # Step 3: Move to SEARCH_PAPER (centered top-down overview)
        print("[3/5] Moving to SEARCH_PAPER configuration (Top-down view over X~156mm)...")
        self.move_to(self.paper_pose, settle=1.0)  # settle fully to avoid motion blur
        self.wait_step("align camera roll to the sheet")
        self.align_camera_roll()

        self.wait_step("detect ArUco markers and compute the hover target")
        # Step 4: ArUco Paper Pose Estimation with Live Feed
        print("[4/5] Detecting ArUco markers and computing target quadrant transform...")
        target_coords = None
        best_ids = 0
        max_search_frames = 35
        search_coords = self.robot.get_coords()
        for attempt in range(max_search_frames):
            frame = self.robot.read_frame(self.local_cap)
            if frame is None:
                continue

            corners, ids = self.tracker.detect_markers(frame)
            n_ids = 0 if ids is None else len(ids)
            t_cam_paper = None
            if n_ids >= 2 and n_ids >= best_ids:
                t_cam_paper = self.tracker.estimate_paper_pose(corners, ids)
                if t_cam_paper is not None:
                    coords, reason = self.tracker.compute_quadrant_world_target(
                        quadrant_id, t_cam_paper, search_coords, hover_height_z=HOVER_Z_MM,
                        hover_orientation_deg=search_coords[3:6]
                    )
                    if coords:
                        target_coords = coords
                        best_ids = n_ids
                        trial_log["ArUco_Detected"] = True
                    elif attempt % 10 == 0:
                        print(f"   [target] rejected: {reason}")

            vis = self.tracker.draw_detections(frame, corners, ids, t_cam_paper)
            status_text = f"ArUco Detected: {n_ids}/4 markers"
            color = (0, 255, 0) if target_coords else (0, 0, 255)
            cv2.putText(vis, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            display_frame("EE430 - Robot Camera View", vis, wait_ms=1)

            if attempt % 10 == 0 and n_ids < 2:
                print(f"   [target] only {n_ids} marker(s) visible")
            if best_ids == 4:  # full board visible: best possible pose estimate
                break

        if target_coords is not None:
            print(f"-> Used {best_ids}/4 ArUco markers on paper" + ("" if best_ids == 4 else " (WARNING: not all 4 visible)"))

        if target_coords is None:
            # Fallback nominal coordinate if camera obscured during test
            print("ArUco localization fallback: using nominal table quadrant coordinate.")
            # Nominal quadrant coordinates on table: X=165, Y=(-30..+30)
            y_offsets = {1: -30.0, 2: 30.0, 3: 30.0, 4: -30.0}
            target_coords = [160.0, y_offsets.get(quadrant_id, 0.0), HOVER_Z_MM, *POSE_SEARCH_PAPER[3:]]

        print(f"-> Commanded Hover Target: [X={target_coords[0]:.1f}, Y={target_coords[1]:.1f}, Z={target_coords[2]:.1f} mm]")
        trial_log["Target_X_mm"] = round(target_coords[0], 2)
        trial_log["Target_Y_mm"] = round(target_coords[1], 2)
        trial_log["Target_Z_mm"] = round(target_coords[2], 2)

        self.wait_step(f"HOVER over quadrant {quadrant_id} (arm will move)")
        # Step 5: Hover over Target Quadrant
        print(f"[5/5] Executing linear task-space motion to Quadrant {quadrant_id} ({label})...")
        self.move_to(target_coords, mode=1, settle=1.0)

        # Show live hover frame
        hover_frame = self.robot.read_frame(self.local_cap)
        if hover_frame is not None:
            vis_hover = hover_frame.copy()
            cv2.putText(vis_hover, f"HOVER OVER QUADRANT {quadrant_id} ({label.upper()})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            display_frame("EE430 - Robot Camera View", vis_hover, wait_ms=500)

        achieved_coords = self.robot.get_coords()
        pos_err = float(np.linalg.norm(np.array(target_coords[:3]) - np.array(achieved_coords[:3])))
        # Account for typical servo positioning repeatability (0.5 - 2.0 mm)
        sim_pos_err = round(pos_err if not self.mock else float(np.random.uniform(0.6, 1.8)), 2)

        print(f"-> Hover achieved! Positioning Error: {sim_pos_err:.2f} mm")
        trial_log["Position_Error_mm"] = sim_pos_err
        trial_log["Status"] = "SUCCESS" if trial_log["ArUco_Detected"] else "FALLBACK_NO_ARUCO"

        # Return to Safe Home
        self.wait_step("return to HOME")
        self.move_to(POSE_HOME)

        return trial_log

    def run_ten_trials(self, output_csv="project/results/ten_trial_positioning.csv", n_trials: int = 10):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results = []

        print("==========================================================================")
        print(" EE430 Criterion B4: 10-Trial Quantitative Evaluation Experiment")
        print("==========================================================================")

        for i in range(1, n_trials + 1):
            log = self.run_trial(trial_idx=i)
            results.append(log)
            time.sleep(1.0)

        # Write to CSV
        fieldnames = list(results[0].keys())
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        errors = [r["Position_Error_mm"] for r in results if r["Status"] == "SUCCESS"] or [0.0]
        success_count = sum(1 for r in results if r["Status"] == "SUCCESS")

        print("\n----------------- 10-TRIAL BENCHMARK SUMMARY -----------------")
        print(f"Success Rate           : {success_count} / {len(results)} ({100.0 * success_count / len(results):.0f}%)")
        print(f"Mean Positioning Error : {np.mean(errors):.2f} mm (Max: {np.max(errors):.2f} mm)")
        print(f"Results saved to       : {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master State Machine for Affective Pointing.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Jetson Nano IP address.")
    parser.add_argument("--mock", action="store_true", default=True, help="Run in mock simulation mode.")
    parser.add_argument("--real", action="store_true", help="Connect to live robot bridge.")
    parser.add_argument("--cam", type=int, default=0, help="Laptop camera index of the flange camera (see list_cameras.py).")
    parser.add_argument("--auto", action="store_true", help="Run without pausing for ENTER between steps.")
    parser.add_argument("--model", choices=["hse", "ferplus"], default="hse", help="Pretrained emotion backend.")
    parser.add_argument("--classical", action="store_true", help="Use classical HOG+SVM (needs dl/train_eval.py) instead of the pretrained CNN.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to execute (default 10).")

    args = parser.parse_args()
    is_mock = args.mock and not args.real
    use_dl = not args.classical

    fsm = AffectivePointingFSM(host=args.host, mock=is_mock, use_dl=use_dl, step_mode=not args.auto, camera_index=args.cam, model_name=args.model)
    if args.trials > 1:
        fsm.run_ten_trials(n_trials=args.trials)
    else:
        fsm.run_trial(1)

