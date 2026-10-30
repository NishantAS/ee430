# EE430 Course Project: Vision-Guided Affective Manipulator (myCobot 280 Jetson Nano)

An autonomous affective robotics pipeline connecting Spong Standard Denavit-Hartenberg (DH) kinematics, Eye-in-Hand camera calibration, ArUco fiducial paper localization, and facial expression recognition (Classical HOG+SVM & MobileNetV2 Deep Learning Bonus) on the 6-DOF **myCobot 280 Jetson Nano**.

---

## 1. System Architecture & Workflow

Due to lab sharing rules, this codebase implements a **Zero-Footprint Remote Client–Server Architecture**:
1. **Jetson Nano (Edge Node):** Runs `robot_bridge.py` (copied via SCP during lab slot and cleaned up before return). Streams flange camera video (Port 5000) and receives JSON joint/task-space RPC commands (Port 5001).
2. **Laptop Master Controller:** Executes the state machine in `main.py`, computes Spong DH Forward Kinematics, solves ArUco pose estimation (`cv2.solvePnP`), and runs facial emotion recognition on laptop GPU.

```
 [Flange Camera] ---> /dev/video0 ---> robot_bridge.py (Jetson Nano)
                                             | TCP Stream (Port 5000/5001)
                                             v
 [Laptop FSM] --------> Emotion Pipeline (Classical vs DL)
              --------> ArUco Board solvePnP (^C T_M)
              --------> Spong DH Kinematics (^0 T_6 * ^6 T_C * ^C T_M * ^M p_Q)
              --------> mc.send_coords() -> Robot arm hovers above target quadrant
```

---

## 2. Directory Structure

```text
project/
├── README.md                        # Complete operational and reproduction guide
├── requirements.txt                 # Dependencies (numpy, opencv, torch, pymycobot, sklearn)
├── main.py                          # Master Finite State Machine (runs full closed-loop demo)
├── robot_bridge.py                  # Jetson Nano lightweight video & RPC server
├── robot_client.py                  # Laptop TCP client to bridge server
├── assets/
│   └── aruco_target_sheet_A4.png    # Printable 300 DPI A4 target sheet (4 ArUco corners + 4 quadrants)
├── kinematics/                      # Component A (4%)
│   ├── dh_model.py                  # Spong Standard DH parameter table & FK implementation
│   ├── fk_verification.py           # Evaluates 15 test poses -> results/fk_error_table.csv
│   └── ik_workspace_analysis.py     # Multi-solution & singularity failure case study
├── vision/                          # Component B (4%)
│   ├── aruco_board.py               # Generates printable A4 sheet with 4 corner markers
│   ├── aruco_tracker.py             # CLAHE detection, solvePnP, target quadrant transform
│   └── classical_emotion.py         # Baseline Haar Cascade + HOG (1764-d) + Linear Classifier
├── dl/                              # Component E Bonus (+4%)
│   ├── dataset_collector.py         # Interactive lab face dataset collector (train/test splits)
│   ├── dl_emotion_infer.py          # MobileNetV2 transfer learning emotion inference
│   └── train_eval.py                # Quantitative accuracy, latency, and parameter comparison
├── calibration/                     # Calibration utilities
│   ├── calibrate_intrinsics.py      # 9x6 checkerboard intrinsic calibration
│   ├── hand_eye_calibration.py      # Flange-to-camera offset verification (^6 T_C)
│   └── camera_params.npz            # Intrinsic K, distortion D, and ^6 T_C matrix
└── results/                         # Quantitative logs for IEEE paper
    ├── fk_error_table.csv           # Criterion A2: 15-pose error table
    ├── ik_case_study.csv            # Criterion A3: multi-solution & failure cases
    ├── ten_trial_positioning.csv    # Criterion B4: 10-trial positioning log
    └── dl_vs_classical_metrics.csv  # Criterion E: DL vs Classical benchmark
```

---

## 3. Installation & Setup

Activate your Python virtual environment and install required packages:
```bash
source .myenv/bin/activate
pip install -r project/requirements.txt
```

---

## 4. Running the System

### 4.1 Mock Simulation Mode (No Robot Hardware Needed)
You can test the entire closed-loop state machine right on your laptop:
```bash
# Run full 10-trial benchmark with simulated robot and local camera
python project/main.py --mock --trials 10

# Run Spong DH Forward Kinematics verification:
python project/kinematics/fk_verification.py --mock

# Run IK workspace & multi-solution case study:
python project/kinematics/ik_workspace_analysis.py --mock

# Run DL vs Classical benchmark evaluation:
python project/dl/train_eval.py
```

### 4.2 Printable A4 Target Sheet
Generate the printable 300 DPI A4 target sheet:
```bash
python project/vision/aruco_board.py
```
Print `project/assets/aruco_target_sheet_A4.png` on standard A4 paper with 100% scale (no fit-to-page). Place it flat on the lab table within:
- $X \in [145\text{ mm}, 185\text{ mm}]$
- $Y \in [-40\text{ mm}, +40\text{ mm}]$

---

## 5. Lab Session Workflow (Jetson Nano Deployment)

### Step 1: Issue & SSH Setup (2:00 PM)
1. Fill out the Robot Issue Google Form and upload the pre-session photo.
2. Connect Ethernet cable between laptop and myCobot Jetson Nano.
3. Copy the bridge script to the Jetson Nano:
   ```bash
   scp project/robot_bridge.py nano@192.168.1.xxx:/tmp/robot_bridge.py
   ```
4. SSH into the Jetson Nano and start the bridge:
   ```bash
   ssh nano@192.168.1.xxx
   python3 /tmp/robot_bridge.py --port /dev/ttyTHS1 --cam 0
   ```

### Step 2: Run Live Master Demo on Laptop
On your laptop:
```bash
python project/main.py --host 192.168.1.xxx --real --trials 10
```

### Step 3: Cleanup & Return (4:15 PM)
1. Stop `robot_bridge.py` on the Jetson Nano (`Ctrl+C`).
2. Delete `/tmp/robot_bridge.py` on the Jetson Nano:
   ```bash
   rm -f /tmp/robot_bridge.py
   ```
3. Command arm to home/rest pose, fill out the Robot Return Google Form, and upload the return photo.

