# EE430 Course Project: Project Context & Master Reference

## 1. Executive Summary & Objective

* **Course:** EE430: Introduction to Robot Dynamics and Control.
* **Platform:** **myCobot 280 Jetson Nano** (6-DOF articulated serial robotic arm + Elephant Robotics USB Flange Camera).
* **Task:** **Closed-Loop Vision-Guided Affective Manipulator**:
  1. The robot looks up at the user (`SEARCH_FACE`) and classifies their facial expression (Happy, Angry, Surprise, Neutral) using Classical Vision (Haar+HOG) and Deep Learning (MobileNetV2).
  2. The robot transitions to a top-down overview (`SEARCH_PAPER`) over the lab table.
  3. The arm detects an A4 sheet with 4 corner ArUco markers (`DICT_4X4_50`, IDs 0–3) using CLAHE and `cv2.solvePnP` to estimate the 6-DOF paper pose (${}^{C}T_{M}$).
  4. The robot calculates the target quadrant center in the robot base frame:
     $${}^{0}\mathbf{p}_{Q_i} = {}^{0}T_{6}(\mathbf{q}) \cdot {}^{6}T_{C} \cdot {}^{C}T_{M} \cdot {}^{M}\mathbf{p}_{Q_i}$$
  5. The arm executes linear task-space motion (`send_coords`, mode=1) to hover directly over the selected quadrant at flange height $Z = 130\text{ mm}$ with a downward-pointing camera.

---

## 2. Evaluation & Grading Criteria Mapping

| Component | Weight | Key Deliverables & Implementation |
| :--- | :--- | :--- |
| **A. Kinematics** | 4% | • **A1:** Repeatable execution in live demo.<br>• **A2:** Spong Standard DH model in `kinematics/dh_model.py` verified across 15 poses vs `mc.get_coords()` in `results/fk_error_table.csv` ($e_{\text{pos}} < 1.0\text{ mm}$).<br>• **A3:** Multi-solution (Elbow-Up vs Elbow-Down) and boundary singularity analysis in `results/ik_case_study.csv`.<br>• **A4:** Strict SI/mm/degree frame handling and workspace safety guardrails ($R \in [110, 235]\text{ mm}, Z \ge 70\text{ mm}$). |
| **B. Vision** | 4% | • **B1:** CLAHE-enhanced ArUco detection robust under lab lighting.<br>• **B2:** Eye-in-Hand camera calibration (${}^{6}T_{C}$) in `calibration/`.<br>• **B3:** Closed-loop vision feedback driving task-space hover target.<br>• **B4:** 10-trial positioning error & accuracy logged in `results/ten_trial_positioning.csv`. |
| **C. Report** | 8% | 8–12 page IEEE conference paper detailing DH derivation, vision pipelines, experimental tables, and discussion of limitations. |
| **D. Viva** | 4% | Individual viva defense and code walkthrough. |
| **E. DL Bonus** | +4% | MobileNetV2 transfer learning vs Classical Haar+HOG baseline comparison (Accuracy %, Latency ms, FPS) in `results/dl_vs_classical_metrics.csv`. |

---

## 3. Network & System Architecture

Because the robot is shared across groups in the lab and must remain clean between sessions, the system uses a **Zero-Footprint Remote Client–Server Architecture**:

```
+---------------------------------------------------------------------------------+
|                             USER LAPTOP (Master Node)                           |
|                                                                                 |
|   +-----------------------+        +----------------------------------------+   |
|   |    TCP Video Client   | -----> |     Master State Machine (main.py)     |   |
|   +-----------------------+        +----------------------------------------+   |
|                                         |                       |               |
|                    (State: SEARCH_FACE) v                       v (SEARCH_PAPER)|
|              +--------------------------+          +------------------------+   |
|              | Emotion Recognition:     |          | ArUco Localization:    |   |
|              | - Classical: Haar + HOG  |          | - cv2.solvePnP multi-  |   |
|              | - DL: MobileNetV2 Head   |          |   marker board pose    |   |
|              +--------------------------+          +------------------------+   |
|                           | (Quadrant 1..4)                     | (^C T_M)      |
|                           +---------------->  +--------------+  <---------------+   |
|                                               | Kinematics:  |                      |
|                                               | ^0 T_Q = FK  |                      |
|                                               +--------------+                      |
|                                                       | (Target X, Y, Z, Rx, Ry, Rz)|
|                                                       v                             |
|                                         +---------------------------+               |
|                                         | TCP RPC Command Client    |               |
|                                         +---------------------------+               |
+-------------------------------------------------------|-------------------------+
                                                        | TCP Socket (Port 5000 & 5001)
                                                        v
+---------------------------------------------------------------------------------+
|                       MYCOBOT 280 JETSON NANO (Edge Node: 10.42.0.1)            |
|                                                                                 |
|   +-----------------------------+           +-------------------------------+   |
|   | OpenCV Video Streamer       |           | pymycobot RPC Listener        |   |
|   | - Port 5000: JPEG stream    |           | - Port 5001: JSON Commands    |   |
|   +-----------------------------+           +-------------------------------+   |
|                  | (/dev/video0 / 1)                        | (/dev/ttyTHS1 UART)   |
|                  v                                          v                       |
|       [Elephant Flange Camera]                     [6x Servo Actuators]             |
+---------------------------------------------------------------------------------+
```

---

## 4. Hardware Calibration & Operational Ground Truth

Through real-world physical robot testing in the lab, the following empirical configurations and offsets have been established:

### 4.1 Calibrated Base / Home Pose
* **Empirical Angle Reading:**
  ```python
  POSE_HOME = [0.43, -0.26, -0.43, -0.26, -0.26, 52.48]
  ```
* **Critical Finding:** The Elephant Robotics Flange Camera has a physical mounting offset of **52.48° on Joint 6** (12.48° in the DH model plus a measured 40° roll) for a level image. This offset is baked into all poses.

### 4.2 Top-Down Paper Search Pose
* **Empirical Joint Angles:**
  ```python
  POSE_SEARCH_PAPER = [162.0, -26.63, 270.0, 180.0, 0.0, -130.48]  # coordinates (camera ~245 mm up); HOVER_Z_MM = 130
  ```
* **Kinematic Result in Base Frame:**
  * **Flange Pose:** $X = 162.0\text{ mm}, Y = -26.6\text{ mm}, Z = 270.0\text{ mm}$
  * **Camera Optical Position:** $X \approx 156\text{ mm}, Y \approx 3\text{ mm}, Z \approx 245\text{ mm}$
  * **Camera Optical Axis ($Z_C$):** $[0.0, 0.0, -1.0]$ (pointing **straight vertically down** at table center).

### 4.3 Human Face Search Pose
* **Empirical Joint Angles:**
  ```python
  POSE_SEARCH_FACE = [0.0, 20.0, -40.0, 20.0, 0.0, 52.48]
  ```
* **Kinematic Result:**
  * **Camera Height:** $Z \approx 398\text{ mm}$
  * **Camera Optical Axis:** $[1.0, 0.0, 0.0]$ (pointing **horizontally forward** directly at the human face).

### 4.4 Eye-in-Hand Transformation Matrix (${}^{6}T_{C}$)
* Camera optical axis parallel to flange, with housing translation:
  $$t_{6 \to C} = [-30.0\text{ mm}, 0.0\text{ mm}, 25.0\text{ mm}]$$

---

## 5. Physical Target Sheet Layout (A4, 300 DPI)

* **Asset:** `project/assets/aruco_target_sheet_A4.png`
* **Active Grid:** $120\text{ mm} \times 120\text{ mm}$ divided into 4 quadrants ($60\text{ mm} \times 60\text{ mm}$ each).
* **Corner ArUco Markers (`DICT_4X4_50`, 28 mm size):**
  * `ID 0`: $(-75\text{ mm}, +75\text{ mm})$ [Top-Left]
  * `ID 1`: $(+75\text{ mm}, +75\text{ mm})$ [Top-Right]
  * `ID 2`: $(+75\text{ mm}, -75\text{ mm})$ [Bottom-Right]
  * `ID 3`: $(-75\text{ mm}, -75\text{ mm})$ [Bottom-Left]
* **Quadrant Target Centers in Paper Frame $\{M\}$:**
  * `Quadrant 1 (Happy)`: $(-30\text{ mm}, +30\text{ mm}, 0\text{ mm})$
  * `Quadrant 2 (Angry)`: $(+30\text{ mm}, +30\text{ mm}, 0\text{ mm})$
  * `Quadrant 3 (Surprise)`: $(+30\text{ mm}, -30\text{ mm}, 0\text{ mm})$
  * `Quadrant 4 (Neutral)`: $(-30\text{ mm}, -30\text{ mm}, 0\text{ mm})$
* **Placement Zone on Table:** $X \in [145\text{ mm}, 185\text{ mm}]$, $Y \in [-40\text{ mm}, +40\text{ mm}]$.

---

## 6. Repository Structure

```text
project/
├── context.md                       # This master context reference document
├── plan.md                          # Comprehensive master engineering plan
├── README.md                        # Submission and reproduction guide
├── requirements.txt                 # Dependencies: opencv, torch, pymycobot, sklearn
├── main.py                          # Master Finite State Machine (7-state closed loop)
├── robot_bridge.py                  # Jetson Nano edge server (TCP ports 5000 & 5001)
├── robot_client.py                  # Laptop RPC client (commands + video decoding)
├── reset_home.py                    # One-click utility to return arm to calibrated base pose
├── adjust_pose.py                   # Interactive tool to read/adjust the end-effector pose
├── assets/
│   ├── aruco_target_sheet_A4.png    # Printable 300 DPI A4 target sheet
│   └── haarcascade_frontalface_default.xml # Bundled offline face detector cascade
├── kinematics/
│   ├── dh_model.py                  # Spong Standard DH parameter model & FK
│   ├── fk_verification.py           # 15-pose evaluation script -> results/fk_error_table.csv
│   └── ik_workspace_analysis.py     # Multi-solution & failure cases -> results/ik_case_study.csv
├── vision/
│   ├── aruco_board.py               # Generates printable A4 sheet
│   ├── aruco_tracker.py             # CLAHE detection, solvePnP, target quadrant transform
│   └── classical_emotion.py         # Baseline Haar Cascade + HOG (1764-d) + Linear Classifier
├── dl/
│   ├── dataset_collector.py         # Interactive lab face dataset collector (train/test splits)
│   ├── dl_emotion_infer.py          # MobileNetV2 GPU/CPU emotion inference wrapper
│   └── train_eval.py                # Quantitative benchmark evaluation
├── calibration/
│   ├── calibrate_intrinsics.py      # 9x6 checkerboard intrinsic calibration
│   ├── hand_eye_calibration.py      # Flange-to-camera offset verification (^6 T_C)
│   └── camera_params.npz            # Stored intrinsic matrix K, D, and ^6 T_C
└── results/
    ├── fk_error_table.csv           # Criterion A2: 15-pose FK error table (mean < 0.6 mm)
    ├── ik_case_study.csv            # Criterion A3: Multi-solution & singularity log
    ├── ten_trial_positioning.csv    # Criterion B4: 10-trial positioning accuracy log
    └── dl_vs_classical_metrics.csv  # Criterion E: DL vs Classical benchmark metrics
```

---

## 7. Quick Operational Cheat Sheet

### Setting up on Jetson Nano (at start of lab slot):
```bash
# 1. Copy bridge to Jetson
scp project/robot_bridge.py nano@10.42.0.1:/tmp/robot_bridge.py

# 2. SSH into Jetson and start server
ssh nano@10.42.0.1
python3 /tmp/robot_bridge.py --port /dev/ttyTHS1 --baud 1000000
```

### Running from Laptop:
```bash
# Reset arm to calibrated base pose:
python project/reset_home.py --host 10.42.0.1

# Inspect or adjust the end-effector pose:
python project/adjust_pose.py --host 10.42.0.1
python project/adjust_pose.py --host 10.42.0.1 --axis rz --value -120

# Run live closed-loop master demo (1 trial or 10 trials):
python project/list_cameras.py   # find the flange camera index first
python project/main.py --host 10.42.0.1 --real --cam 2 --trials 1
python project/main.py --host 10.42.0.1 --real --cam 2 --trials 10

# Run offline in mock mode (no hardware required):
python project/main.py --mock --trials 10
```

### Cleaning up Jetson Nano (at end of lab slot):
```bash
# Return arm to home pose, stop bridge (Ctrl+C), then delete:
rm -f /tmp/robot_bridge.py
```

