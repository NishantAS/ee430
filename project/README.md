# EE430 – Gesture-commanded myCobot 280 (Jetson Nano)

Show **1–4 fingers** to the camera on the robot's flange → the arm hovers over that quadrant of a printed
ArUco target sheet.

| Rubric | What does it | Command |
|---|---|---|
| A1 task | gesture → quadrant hover, repeatable | `main.py` |
| A2 FK vs own DH | 15 poses, our FK vs `get_coords()` | `experiments.py fk` |
| A3 IK | one pose, two joint solutions (elbow up / V); robot's built-in IK (`send_coords`) on reachable + unreachable targets | `experiments.py ik` |
| A4 frames/units | mm + degrees everywhere, one base frame, `config.py` | – |
| B1 detection | ArUco sheet + classical finger counter | `vision/` |
| B2 calibration | pixel → robot-base homography from touched points | `calibrate.py table` |
| B3 closed loop | vision output → `send_coords` (verified with `get_coords`) | `main.py` |
| B4 ≥ 10 trials | logged to `results/trials.csv` | `main.py --trials 10 --log` |
| E bonus | MediaPipe hand-landmark CNN vs classical, accuracy + latency | `experiments.py gestures` |

## How it works

```
 flange camera (USB → laptop) ──► finger count (MediaPipe | classical) ──► quadrant 1-4
                               └─► ArUco markers ──► H_sheet→pixel ─┐
                     calibration.json: H_pixel→base, table z ───────┴─► quadrant (x, y) in base frame
 target coords (tool straight down) ──► TCP ──► robot_bridge.py on Jetson ──► pymycobot send_coords
```

* **Motion** uses pymycobot only: `send_angles` for fixed poses (home, overview: exactly repeatable) and
  `send_coords` for task-space moves (the robot solves the IK). `send_coords` does not report failure, so
  every move is read back with `get_coords` and raises an error if the arm does not arrive (= unreachable).
  The "pointing down" orientation is the robot's own reading at the calibrated overview pose.
* **Our DH model** (`kinematics/dh_model.py`) is used only to verify the robot's FK (A2) and in
  `robot_tools.py status`.
* **Calibration** (`calibrate.py table`): the overview pose is a fixed joint configuration, so the camera
  always sees the table plane the same way, and pixels map to base (x, y) by a homography. It is fitted
  from the 4 marker centres (seen in the image + touched with the tool, positions from `get_coords`) and checked
  on a 5th touched point (sheet centre). No camera intrinsics or hand-eye matrix are needed.
* **The sheet can move between trials**: every trial re-detects it; only the overview pose must stay fixed.

## Arm shape: always elbow up (important)

The robot's built-in IK keeps the elbow configuration the arm is already in. From an elbow-down "V" (or a
straight elbow, which can go either way) it reaches down in a V and can hit the base. So:
* every `send_coords` move refuses to start unless the elbow is clearly bent up (q3 ≤ −15°; q3 < 0 is the
  inverted V in our DH convention), and is stopped (`stop()`) if q3 rises above +5° while moving;
* task-space moves are straight lines (`send_coords` mode 1), so the IK follows the path continuously;
* fixed poses (home, ready, overview) are reached by joint angles, and the taught overview must be elbow-up.

If a script says the elbow is not bent up, run `python robot_tools.py ready` (or `overview`).

## Workspace limits (important)

With the tool pointing straight down, the arm only reaches roughly **x = 160–240 mm** from the base (closer
in, the elbow hits its joint limit; the robot then silently ignores `send_coords`). **Put the sheet centre
near x ≈ 200 mm, y ≈ −60 mm** (straight in front of the arm; the −60 comes from the 64.6 mm sideways wrist
offset). `experiments.py ik` tests targets inside and outside this region on the real robot.

## Setup (laptop, once)

```bash
source ../.myenv/bin/activate        # Python ≥ 3.9
pip install -r requirements.txt
python calibrate.py sheet            # → assets/target_sheet_A4.png, print at 100 %, check markers = 28 mm
```
The MediaPipe model (7.8 MB) downloads itself to `assets/` on first use.

## Lab session

**1. Start** (fill the issue form + photo first)
```bash
scp robot_bridge.py nano@10.42.0.1:/tmp/          # user/IP of your robot
ssh nano@10.42.0.1 "python3 /tmp/robot_bridge.py"   # leave running; Ctrl+C to stop
```
Plug the flange camera's USB into the laptop. All laptop commands run from this `project/` folder and
accept `--host <ip>` (default `10.42.0.1`).

```bash
python robot_tools.py status      # connection check: angles, get_coords vs our DH FK
python robot_tools.py ready       # elbow-up start pose - every send_coords move must start elbow-up
```

**2. Calibrate** (first session, and again whenever the robot or table setup changes)
```bash
python calibrate.py camera        # press y on the flange camera's image
python calibrate.py overview      # move the arm by hand until all 4 markers are green, ENTER
python calibrate.py table         # by hand: tool tip onto marker 0,1,2,3 centres + sheet centre
python calibrate.py check         # arm visits all 4 quadrant crosses - verify by eye / ruler
# or all four in sequence:  python calibrate.py all
```
Teaching by hand (camera window focused): **hold the arm**, `SPACE` releases the servos (the arm goes limp),
move it (keep the elbow bent up), `ENTER` locks it exactly where it is and records the pose, `ESC` aborts. The arm is always locked
again before the script continues or exits. For the overview the robot then straightens the camera to point
straight down; `ENTER` saves, `SPACE` teaches again. For the touches keep the tool roughly vertical. If you tape
a pointer to the flange, set `TOOL_LENGTH_MM` in `config.py` first.

**3. Experiments and demo**
```bash
python experiments.py fk                         # A2
python experiments.py ik                         # A3 (needs calibrate.py table: uses the table height)
python robot_tools.py overview && python experiments.py collect   # label gesture images with keys 0-5
python experiments.py gestures                   # classical vs MediaPipe → results/gesture_metrics.csv
python main.py --trials 10 --log                 # B4 → results/trials.csv + photos; measure the miss with a ruler
python main.py --classifier classical --trials 10 --log
```
In `main.py`, `q`/`ESC` stops; keys `1-4` override the gesture (logged as `keyboard`).

**4. Finish**
```bash
python robot_tools.py home
ssh nano@10.42.0.1 "rm -f /tmp/robot_bridge.py"     # after Ctrl+C on the bridge
```
Then the return form + photo.

## Without the robot

`--mock` replaces the arm with an instant-moving simulation (camera still real, e.g. a webcam with
`--cam 0`). It reaches every target, so it says nothing about reachability. Anything it writes to
`results/` is prefixed `MOCK_`, so it cannot be mixed up with lab data.

## Files

```
config.py            every constant (poses, limits, heights, paths)  ← tune here
main.py              demo
calibrate.py         calibration steps → calibration/calibration.json
experiments.py       report tables → results/
robot_tools.py       status / home / overview / goto / release / lock
robot_bridge.py      runs on the Jetson (Python 3.6 compatible)
common/              robot_client (TCP + mock), camera, teach (hand-guiding), calibration_store
kinematics/          dh_model: our DH forward kinematics (A2 verification only)
vision/              sheet (ArUco + homographies + printable), gesture_classical, gestures (factory)
dl/                  gesture_mediapipe (hand-landmark CNN)
```

Third-party code: OpenCV (ArUco, homography), Google MediaPipe Hand Landmarker (pretrained model),
pymycobot (on the robot). Cite them in the report.
