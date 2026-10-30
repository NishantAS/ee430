All requirements are completely clear—no further questions needed. Below is your comprehensive, step-by-step **Master Engineering & Implementation Plan** designed specifically around the **myCobot 280 Jetson Nano** hardware, your SSH/laptop-GPU architecture, and every single grading rubric item (A1–A4, B1–B4, C1–C5, D, and Bonus E).

---

## 1. End-to-End System Architecture

### 1.1 Physical & Network Topology

Because the robot is shared across groups and all code must reside on your laptop between sessions, your system uses a **Zero-Footprint Remote Client–Server Architecture**:

1. **Jetson Nano (Edge Server):**
* Connected to the 6-DOF arm servos via internal UART (`/dev/ttyTHS1` at `1000000` baud) and to the Elephant Robotics USB Flange Camera (`/dev/video0`).
* Connected to your laptop via SSH over Wired Ethernet (or lab Wi-Fi).


* Runs a single temporary script (`robot_bridge.py`) copied over `scp` at the start of the lab slot and deleted before filling out the return form.




2. **User Laptop (Master Controller & GPU Inference Node):**
* Receives compressed JPEG frames (`640x480` @ 30 FPS) over a TCP socket from `robot_bridge.py`.
* Executes the Finite State Machine (`main.py`), ArUco pose estimation, Classical vs. Deep Learning facial expression pipelines, and Spong DH Forward Kinematics verification.


* Sends joint-space (`send_angles`) and task-space (`send_coords`) commands back over TCP to the Jetson Nano.





```
+-----------------------------------------------------------------------------------+
|                               USER LAPTOP (PC Master)                             |
|                                                                                   |
|  +-------------------+      +--------------------------------------------------+  |
|  | TCP Frame Client  | ---> |           Finite State Machine (main.py)         |  |
|  +-------------------+      +--------------------------------------------------+  |
|                                |                      |                       |   |
|         +----------------------+                      |                       |   |
|         v (State: CAPTURE_FACE)                       v (State: LOCATE_PAPER) |   |
|  +------------------------------+      +-------------------------------+      |   |
|  | Emotion Recognition Pipeline |      | ArUco Localization Pipeline   |      |   |
|  | - Mode 1: Classical (Haar +  |      | - cv2.aruco (DICT_4X4_50)     |      |   |
|  |   HOG/SVM or Mouth/Eye Ratio)|      | - solvePnP -> ^C T_M          |      |   |
|  | - Mode 2: DL (HSEmotion /    |      | - Quadrant Offset -> ^M T_Qi  |      |   |
|  |   MobileNetV2 on Laptop GPU) |      +-------------------------------+      |   |
|  +------------------------------+                     |                       |   |
|         | (Emotion ID: 1..4)                          v (Target ^0 T_Qi)      |   |
|         +----------------------> +-------------------------------------+ <----+   |
|                                  | Kinematics Module (kinematics/)     |          |
|                                  | - Spong DH FK Verification (^0 T_6) |          |
|                                  | - Hand-Eye Transform (^6 T_C)       |          |
|                                  | - Workspace & Singularity Validator |          |
|                                  +-------------------------------------+          |
|                                                       |                           |
|                                                       v                           |
|                                        +-------------------------------+          |
|                                        | TCP RPC Command Client        |          |
|                                        +-------------------------------+          |
+-------------------------------------------------------|---------------------------+
                                                        | SSH / TCP Socket
                                                        v (Port 5000: Video, 5001: Cmd)
+-----------------------------------------------------------------------------------+
|                          MYCOBOT 280 JETSON NANO (Robot Edge)                     |
|                                                                                   |
|  +------------------------------+             +--------------------------------+  |
|  | OpenCV Video Streamer        |             | pymycobot RPC Listener         |  |
|  | - Captures /dev/video0       |             | - mc.send_angles(angles, spd)  |  |
|  | - JPEG Encode (Quality 80)   |             | - mc.send_coords(coords, spd)  |  |
|  +------------------------------+             | - mc.get_angles() / get_coords |  |
|                 ^                             +--------------------------------+  |
|                 | USB 2.0                                     | UART /dev/ttyTHS1 |
|     [Elephant Flange Camera]                         [6x Servo Actuators]         |
+-----------------------------------------------------------------------------------+

```

---

## 2. Repository & Directory Structure

Set up your laptop repository from Day 1 using the exact archive structure required by Page 4 of the instructions so you never have to refactor before submission:

```text
EE430_GroupNN_Code/
├── README.md                        # Step-by-step setup, SSH deployment, and model links
├── requirements.txt                 # numpy, opencv-contrib-python, torch, torchvision, pymycobot, scikit-learn
├── main.py                          # Master Finite State Machine (runs the full demo task)
├── robot_bridge.py                  # Lightweight TCP server copied to Jetson Nano via SCP
├── kinematics/                      # Component A (4%)
│   ├── __init__.py
│   ├── dh_model.py                  # Spong convention DH table, A_i matrices, and FK computation
│   ├── fk_verification.py           # Logs 15 test poses comparing dh_model.py vs pymycobot.get_coords()
│   └── ik_workspace_analysis.py     # Tests pymycobot.send_coords() across valid, singular, & out-of-reach poses
├── vision/                          # Component B (4%)
│   ├── __init__.py
│   ├── aruco_board.py               # Generates printable A4 PDF/PNG with 4 ArUco corners & 4 quadrants
│   ├── aruco_tracker.py             # Detects markers, computes paper pose & target quadrant (X, Y, Z)
│   └── classical_emotion.py         # ~20-line Haar Cascade + HOG/SVM (or Geometric Ratio) baseline classifier
├── dl/                              # Component E Bonus (+4%)
│   ├── __init__.py
│   ├── dataset_collector.py         # Captures lab faces to fine-tune/evaluate expression models
│   ├── train_eval.py                # Fine-tunes lightweight CNN & generates Classical vs DL comparison metrics
│   └── dl_emotion_infer.py          # GPU inference wrapper for real-time emotion classification
├── calibration/                     # Component B2
│   ├── calibrate_intrinsics.py      # Checkerboard calibration -> saves K and dist_coeffs to camera_params.npz
│   ├── hand_eye_calibration.py      # Computes 2D/3D offset (^6 T_C) from flange to Elephant USB camera lens
│   └── camera_params.npz            # Stored intrinsic matrix K, distortion vector D, and ^6 T_C matrix
└── results/                         # Quantitative logs for Section 6 of IEEE Report
    ├── fk_error_table.csv           # Data for Criterion A2
    ├── ik_case_study.csv            # Data for Criterion A3
    ├── ten_trial_positioning.csv    # Data for Criterion B4 (10 trials)
    └── dl_vs_classical_metrics.csv  # Data for Criterion E (Accuracy & FPS comparison)

```

---

## 3. Physical Target Sheet & Workspace Engineering

### 3.1 Solving the Reach Envelope Constraint

* The myCobot 280 has a nominal reach radius of $280\text{ mm}$, an inner dead-zone of $R_{\min} \approx 95\text{ mm}$ (where the arm collides with its own base), and an effective vertical hover ceiling where `send_coords()` can maintain a strictly vertical downward camera orientation ($R_x = 180^\circ, R_y = 0^\circ$) within $R \in [115\text{ mm}, 235\text{ mm}]$.
* If you spread the 4 quadrants to the edges of an A4 page ($210 \times 297\text{ mm}$), at least two quadrants will violate the $235\text{ mm}$ vertical-orientation envelope, causing `pymycobot.send_coords()` to silently fail or stall.

### 3.2 Exact A4 Sheet Layout (`vision/aruco_board.py`)

Print on a standard **A4 sheet ($210 \times 297\text{ mm}$)**, taped or placed flat on the table in front of the robot base:

* **Active Grid Area:** A centered **$120\text{ mm} \times 120\text{ mm}$ square** divided into four **$60\text{ mm} \times 60\text{ mm}$ quadrants**.
* **ArUco Markers (`cv2.aruco.DICT_4X4_50`):**
* Place four **$28\text{ mm} \times 28\text{ mm}$** markers at the outer corners of the active grid, with their centers located at $(\pm 75\text{ mm}, \pm 75\text{ mm})$ relative to the paper's origin $(0,0)$:
* **ID 0:** Top-Left $(-75\text{ mm}, +75\text{ mm})$
* **ID 1:** Top-Right $(+75\text{ mm}, +75\text{ mm})$
* **ID 2:** Bottom-Right $(+75\text{ mm}, -75\text{ mm})$
* **ID 3:** Bottom-Left $(-75\text{ mm}, -75\text{ mm})$




* **Quadrant Centers (in the Paper Coordinate Frame $\{M\}$):**
* **Quadrant 1 (Happy 😊):** $(-30\text{ mm}, +30\text{ mm}, 0\text{ mm})$
* **Quadrant 2 (Angry 😠):** $(+30\text{ mm}, +30\text{ mm}, 0\text{ mm})$
* **Quadrant 3 (Surprised 😮):** $(+30\text{ mm}, -30\text{ mm}, 0\text{ mm})$
* **Quadrant 4 (Neutral/Sad 😐):** $(-30\text{ mm}, -30\text{ mm}, 0\text{ mm})$


* **Placement Zone on Lab Table:** Place the center of the A4 sheet anywhere within $X \in [145\text{ mm}, 185\text{ mm}]$, $Y \in [-40\text{ mm}, +40\text{ mm}]$, and yaw rotation $\psi \in [-45^\circ, +45^\circ]$. Every single quadrant will remain 100% reachable across all 10 trials!

---

## 4. Component A: Kinematics Implementation Plan (4%)

### 4.1 Spong Convention DH Derivation (`kinematics/dh_model.py` — Criterion A2, A4)

* **Strict Requirement:** You must use the **Standard Denavit-Hartenberg (DH) convention from Spong, Hutchinson & Vidyasagar** (not Craig's Modified DH convention, even though Elephant Robotics documentation sometimes lists Modified DH parameters).


* In Spong's convention, each link transform $A_i$ from frame $\{i-1\}$ to $\{i\}$ is the product of four basic transformations:

$$A_i = \text{Rot}_{z,\theta_i} \cdot \text{Trans}_{z,d_i} \cdot \text{Trans}_{x,a_i} \cdot \text{Rot}_{x,\alpha_i} = \begin{bmatrix} \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\ \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$


* **Physical Link Lengths of myCobot 280 JN (in millimeters):**
* Base to Joint 2 elevation: $d_1 = 131.56\text{ mm}$
* Upper arm link (J2 to J3): $a_2 = -110.4\text{ mm}$
* Forearm link (J3 to J4): $a_3 = -96.0\text{ mm}$
* Wrist 1 offset (J4 to J5): $d_4 = 64.62\text{ mm}$ (or $66.39\text{ mm}$ depending on hardware revision—measure with calipers in your first lab session!)
* Wrist 2 offset (J5 to J6): $d_5 = 73.18\text{ mm}$
* Flange offset (J6 to End-Effector): $d_6 = 48.6\text{ mm}$


* **Spong Standard DH Table to Implement and Verify:**

| Joint $i$ | $\theta_i$ (Variable + Home Offset) | $d_i$ (mm) | $a_i$ (mm) | $\alpha_i$ (rad) |
| --- | --- | --- | --- | --- |
| **1** | $q_1$ | $131.56$ | $0$ | $+\pi/2$ |
| **2** | $q_2 - \pi/2$ | $0$ | $-110.40$ | $0$ |
| **3** | $q_3$ | $0$ | $-96.00$ | $0$ |
| **4** | $q_4 - \pi/2$ | $64.62$ | $0$ | $+\pi/2$ |
| **5** | $q_5 + \pi/2$ | $73.18$ | $0$ | $-\pi/2$ |
| **6** | $q_6$ | $48.60$ | $0$ | $0$ |

* **Extracting Pose for Comparison:**
Compute ${}^{0}T_{6} = A_1 A_2 A_3 A_4 A_5 A_6$. Extract position $(X, Y, Z) = {}^{0}T_{6}[0:3, 3]$ in millimeters and ZYX Euler angles $(R_x, R_y, R_z)$ in degrees from ${}^{0}T_{6}[0:3, 0:3]$ to match the exact format returned by `pymycobot.get_coords()`.
* **Generating the FK Error Table (Criterion A2):**
Write `fk_verification.py` to move the robot through **15 diverse joint configurations** (including `HOME`, `SEARCH_FACE`, `SEARCH_PAPER`, and the 4 quadrant hover poses) using `mc.send_angles()`. At each pose, record:
1. Commanded angles $q_1 \dots q_6$
2. Measured encoder angles `mc.get_angles()`
3. Firmware FK coordinates `mc.get_coords()`
4. Your Spong DH model coordinates `dh_fk(get_angles())`
5. Euclidean position error $e_p = \Vert{}\mathbf{p}_{\text{DH}} - \mathbf{p}_{\text{mycobot}}\Vert{}_2$ (typically $< 1.5\text{ mm}$) and orientation error $e_R$. Save directly to `results/fk_error_table.csv`.





### 4.2 Task-Space IK via `pymycobot.send_coords()` & Failure/Multi-Solution Analysis (`kinematics/ik_workspace_analysis.py` — Criterion A3)

Since you are strictly using `pymycobot.send_coords([x, y, z, rx, ry, rz], speed, mode)` for IK:

* How to satisfy Criterion A3 (*"IK used correctly, with multiple-solution or failure case explained"*) without writing a custom IK solver:


1. **Correct Usage in Demo:** Command `mc.send_coords([X_target, Y_target, Z_hover, 180.0, 0.0, yaw_deg], speed=30, mode=1)` (where `mode=1` enforces linear interpolation in task space) to move from the paper overview pose to the target quadrant hover pose.
2. **Multiple-Solution Demonstration (Experiment 1 in `ik_workspace_analysis.py`):**
For a 6-DOF manipulator with a spherical/offset wrist, up to **8 analytical IK solutions** exist (Shoulder Left/Right, Elbow Up/Down, Wrist Flip/No-Flip). `pymycobot`'s internal IK selects the solution closest to the current joint configuration $\mathbf{q}_{\text{current}}$.
* *Lab Test:* Pre-position the arm in an **Elbow-Up** configuration via `send_angles`, then call `send_coords(P_target)` and record `get_angles()` ($\mathbf{q}_{\text{sol,1}}$). Next, pre-position the arm in an **Elbow-Down / Wrist-Flipped** configuration via `send_angles`, call the *exact same* `send_coords(P_target)`, and record `get_angles()` ($\mathbf{q}_{\text{sol,2}}$). Feed both $\mathbf{q}_{\text{sol,1}}$ and $\mathbf{q}_{\text{sol,2}}$ into your `dh_model.py` FK script to prove in your report that two distinct joint vectors produce the exact same Cartesian pose!


3. **IK Failure Cases Demonstration (Experiment 2 in `ik_workspace_analysis.py`):**
Log and explain the three specific cases where `mc.send_coords()` fails or rejects a command:
* *Case A (Boundary Singularity / Out of Reach):* Commanding $X = 295\text{ mm}, Y = 0, Z = 100\text{ mm}$ ($\Vert{}\mathbf{p}\Vert{} > L_{\max}$). Show that `pymycobot` either returns `-1`/ignores the command or saturates joints.
* *Case B (Inner Base Collision / Self-Intersection Singularity):* Commanding $X = 40\text{ mm}, Y = 0, Z = 100\text{ mm}, R_x = 180^\circ$.
* *Case C (Wrist Alignment Singularity $\theta_5 = 0^\circ$):* When $Z_4$ and $Z_6$ axes become collinear, $q_4$ and $q_6$ become coupled, causing large jerky joint rotations during linear (`mode=1`) task-space moves.
* **Code Guardrail:** Add a function `is_target_reachable(x, y, z)` in `kinematics/dh_model.py` that checks $110\text{ mm} \le \sqrt{x^2 + y^2} \le 235\text{ mm}$ and $z \ge 70\text{ mm}$ *before* calling `mc.send_coords()`.





---

## 5. Component B: Classical Computer Vision & Calibration Plan (4%)

### 5.1 Camera Intrinsic Calibration (`calibration/calibrate_intrinsics.py` — Criterion B2)

1. Display or print a $9 \times 6$ checkerboard ($20\text{ mm}$ square size).
2. Capture 15–20 images of the checkerboard from the Elephant Robotics flange camera at varying angles.
3. Run `cv2.findChessboardCorners()` and `cv2.calibrateCamera()` to obtain the $3 \times 3$ intrinsic matrix $K$ and distortion coefficients $D$. Save to `calibration/camera_params.npz`.

### 5.2 Eye-in-Hand Calibration (`calibration/hand_eye_calibration.py` — Criterion B2)

Because the Elephant Robotics flange camera mounts directly onto the end-effector flange, its optical axis $Z_C$ is parallel to the 6th link axis $Z_6$, with a rigid physical offset:

1. Command the arm to a known top-down pose: $\mathbf{p}_{\text{flange}} = [160, 0, 180, 180, 0, -90]$ (`send_coords`).
2. Place a single ArUco marker on the table directly underneath the flange center (measure with a ruler/plumb line at $X_0 = 160\text{ mm}, Y_0 = 0\text{ mm}, Z_0 = 0\text{ mm}$ in the robot base frame).
3. Run `cv2.aruco.estimatePoseSingleMarkers()` to get the marker position $\mathbf{p}_C = [t_x, t_y, t_z]^T$ in the camera frame.
4. Solve for the constant Eye-in-Hand homogeneous transformation matrix ${}^{6}T_{C}$ (specifically the radial lens offset $dx \approx -25\text{ mm}$ to $-35\text{ mm}$ along the flange housing and camera axis rotation $R_z$).
5. **Verification:** Move the arm to 3 different $(X, Y, Z)$ heights above the same stationary marker and verify that the computed world coordinates ${}^{0}\mathbf{p}_{\text{marker}} = {}^{0}T_{6} \cdot {}^{6}T_{C} \cdot {}^{C}\mathbf{p}_{\text{marker}}$ remain constant within $\pm 2\text{ mm}$.

### 5.3 ArUco Paper & Quadrant Localization (`vision/aruco_tracker.py` — Criterion B1, B3)

To ensure robustness under harsh overhead lab lighting:

1. Convert incoming frame to grayscale and apply **CLAHE** (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) to eliminate glare on the white A4 paper.
2. Detect markers using `cv2.aruco.ArucoDetector(dictionary, parameters)` with `DICT_4X4_50`.
3. **Multi-Marker Board Pose Estimation (`cv2.solvePnP`):**
Define a `cv2.aruco.Board` object containing the exact 3D coordinates of all 4 corner markers (IDs 0, 1, 2, 3) in the paper's local frame $\{M\}$.
* Calling `cv2.aruco.estimatePoseBoard()` (or `cv2.solvePnP` over all detected marker corners simultaneously) uses **all visible markers (whether 1, 2, 3, or 4 are visible)** to compute a single optimal rotation vector `rvec` and translation vector `tvec` (${}^{C}T_{M}$) of the paper's center!


4. **Target Quadrant Calculation:**
Given the detected emotion $i \in \{1, 2, 3, 4\}$, take the known local quadrant coordinates ${}^{M}\mathbf{p}_{Q_i} = [\pm 30, \pm 30, 0, 1]^T$ and transform them into the Robot Base Frame $\{0\}$:

$${}^{0}\mathbf{p}_{Q_i} = {}^{0}T_{6}(\mathbf{q}_{\text{search}}) \cdot {}^{6}T_{C} \cdot {}^{C}T_{M} \cdot {}^{M}\mathbf{p}_{Q_i}$$


5. **Camera-Centered Hover Target:**
Since the goal is for the *camera* (or flange) to point at / hover directly above Quadrant $i$ at height $Z_{\text{hover}} = 100\text{ mm}$, subtract the horizontal camera-to-flange offset so the camera lens aligns directly over $(X_{Q_i}, Y_{Q_i})$ at $Z = 100\text{ mm}, R_x = 180^\circ, R_y = 0^\circ, R_z = -90^\circ$.

### 5.4 The 20-Line Classical Emotion Baseline (`vision/classical_emotion.py` — Criterion B & E Comparison)

To satisfy the rubric requirement of comparing your Deep Learning pipeline against a classical vision pipeline on the exact same task:

1. Use OpenCV's built-in **Haar Cascade** (`haarcascade_frontalface_default.xml`) to crop the face to $64 \times 64$ grayscale.
2. Extract a classical **HOG descriptor** (`cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)`) in 5 lines of code and pass it to a lightweight `scikit-learn` **Linear SVM (`LinearSVC`)** or **Multi-class Logistic Regression** trained on the same 4 emotions.
3. Total implementation is ~20 lines of pure classical computer vision + linear classification, giving you a concrete baseline for **Accuracy (%)** and **Inference Time (ms/frame)** to compare against your Deep Learning model in Section 6 of your report.



---

## 6. Component E: Deep Learning Bonus Plan (+4%)

### 6.1 Model Architecture & Dataset (`dl/dl_emotion_infer.py` & `dl/train_eval.py`)

1. **Model Choice:** Use **`HSEmotion` (`enet_b0_8_best_vgaf`)** or **MobileNetV2 / ResNet-18** pretrained on AffectNet/FER2013, filtered down to your 4 target classes: `0: Happy`, `1: Angry`, `2: Surprise`, `3: Neutral`.
2. **Satisfying the Rubric's Lab Data Rule:**
Page 2 states: *"Train or fine-tune on data you collect in the lab or justify clearly why a pretrained model suffices. Report accuracy on a held-out set, inference time per frame on the Jetson Nano (or your laptop, stated clearly), and a comparison against your classical pipeline from B."*


* **Best Strategy (Guaranteed 4/4 Bonus Marks):** Write `dl/dataset_collector.py` to capture **200 total frames** in the lab across your group members (50 frames per emotion $\times$ 4 emotions, takes less than 5 minutes).
* Split into **120 training images** and **80 held-out test images**.
* Fine-tune the final classification head of your CNN (and train the Classical HOG+SVM baseline) on the 120 lab images, and evaluate both on the 80 held-out test images!
* Log the exact Confusion Matrix, Accuracy (%), and Inference Time (ms/frame on your NVIDIA Laptop GPU vs. CPU) into `results/dl_vs_classical_metrics.csv`.





---

## 7. Step-by-Step Lab Execution Roadmap (By Lab Slot)

Since you only have access to the robot on **Tuesday, Thursday, or Friday from 2:00–4:30 PM**, do 90% of the coding on your laptop *before* entering the lab:

### Phase 0: Pre-Lab Preparation (On Laptop Before Slot 1)

1. Write `vision/aruco_board.py` and print the A4 target sheet + a $9 \times 6$ checkerboard.
2. Write `kinematics/dh_model.py` and unit-test the Spong DH matrices offline.
3. Write `vision/aruco_tracker.py`, `vision/classical_emotion.py`, and `dl/dl_emotion_infer.py` and test them using your laptop's built-in webcam.
4. Write `robot_bridge.py` (Jetson server) and `main.py` (Laptop client with a `--mock-robot` flag so the whole state machine runs on your laptop without hardware).

### Phase 1: Lab Slot 1 (2.5 Hours) — Calibration & Kinematics Verification

1. **2:00 PM:** Fill out the **Robot Issue Google Form** and upload the "before" photo. Connect via SSH and `scp` your `robot_bridge.py` to `/tmp/groupNN/`.


2. **2:15 PM:** Find safe joint angles for `HOME`, `SEARCH_FACE` (horizontal camera looking at human), and `SEARCH_PAPER` (top-down camera at $Z \approx 220\text{ mm}$).
3. **2:45 PM:** Run `kinematics/fk_verification.py` across 15 poses and `kinematics/ik_workspace_analysis.py` (elbow-up vs. elbow-down + out-of-reach tests). Copy `fk_error_table.csv` and `ik_case_study.csv` back to your laptop (Component A complete!).
4. **3:15 PM:** Capture 15 checkerboard images from the flange camera and run `calibrate_intrinsics.py` + `hand_eye_calibration.py` to save `camera_params.npz`.
5. **3:45 PM:** Run `dl/dataset_collector.py` through the flange camera to record 200 face images of your group members under actual lab lighting.


6. **4:15 PM:** Delete `/tmp/groupNN/` on the Jetson Nano, return the arm to rest pose, and submit the **Robot Return Google Form**.



### Phase 2: Lab Slot 2 (2.5 Hours) — Closed-Loop Integration & 10 Quantitative Trials

1. **2:00 PM:** Issue robot form + `scp` code.


2. **2:15 PM:** Run full closed-loop state machine (`main.py`):
* `HOME` $\rightarrow$ `SEARCH_FACE` $\rightarrow$ 10-frame majority vote emotion classification $\rightarrow$ `SEARCH_PAPER` $\rightarrow$ ArUco `solvePnP` $\rightarrow$ `send_coords` hover over target quadrant at $Z = 100\text{ mm}$ $\rightarrow$ `WAIT_INPUT` $\rightarrow$ `HOME`.


3. **3:00 PM:** Execute **Criterion B4 (10 Quantitative Trials)**:


* Across 10 trials (cycling through all 4 emotions and moving/rotating the A4 sheet before each trial), measure:
1. Emotion classification result (Correct/Incorrect)
2. ArUco detection success (Yes/No)
3. Commanded quadrant center $(X_{\text{cmd}}, Y_{\text{cmd}})$ vs. actual physical error (measured in millimeters using a ruler/grid on the paper or directly from the centered camera view in the hover pose!).




4. **4:00 PM:** Record the **$\le 2\text{ minute}$ demo video** required for `results/` and capture frame sequences for Section 6 of the IEEE report. Clean up Jetson Nano and submit return form.



---

## 8. IEEE Report Blueprint (8–12 Pages, Component C — 8%)

Structure your IEEE two-column LaTeX/Word document strictly around the rubric criteria:

1. **Abstract (150–200 words):** State the closed-loop emotion-to-quadrant pointing task on the 6-DOF myCobot 280 JN, the Spong DH + Eye-in-Hand ArUco architecture, and your key quantitative metrics (mean FK error in mm, 10-trial positioning accuracy in mm, and DL vs. Classical accuracy/FPS).


2. **1. Introduction:** Problem statement (human-robot affective interaction and eye-in-hand visual positioning), system constraints (no end-effector, 280 mm reach envelope), and 3 bulleted contributions.


3. **2. Literature Review (Criterion C1 — 1.5 marks):** Cite at least 6 sources ($\ge 3$ IEEE journal/conference papers) covering (a) fiducial marker pose estimation (Garrido-Jurado et al., *ArUco*, Pattern Recognition 2014), (b) eye-in-hand calibration (Tsai & Lenz, *IEEE Trans. Robotics and Automation*), and (c) lightweight facial expression recognition (Savchenko, *HSEmotion*, IEEE FG / MobileNetV2). Include a **comparison table** contrasting their accuracy, computational cost, and suitability for embedded robotics.


4. **3. System Description:** Hardware diagram of myCobot 280 + Elephant USB flange camera, SSH Client-Server TCP streaming architecture diagram, and finite state machine diagram.


5. **4. Methodology (Criterion C2 — 2.5 marks):**
* *4.1 Kinematic Model:* Coordinate frame diagram (Frames $\{0\}$ to $\{6\}$ + $\{C\}$), Spong DH parameter table, homogeneous link matrices $A_1 \dots A_6$, and task-space IK strategy.


* *4.2 Vision & Eye-in-Hand Calibration:* Intrinsic matrix $K$, ${}^{6}T_{C}$ derivation, multi-marker ArUco `solvePnP` transformation chain ${}^{0}\mathbf{p}_{Q_i} = {}^{0}T_{6} {}^{6}T_{C} {}^{C}T_{M} {}^{M}\mathbf{p}_{Q_i}$.


* *4.3 Facial Expression Recognition:* Classical HOG+SVM pipeline vs. Deep Learning CNN architecture, lab dataset collection, and fine-tuning setup.




6. **5. Results (Criterion C3 — 2.0 marks):**
* *Table I:* 15-Pose Forward Kinematics Error Table (Spong DH vs. `pymycobot.get_coords()`).


* *Table II:* IK Multi-Solution (Elbow-Up vs. Elbow-Down) and Out-of-Workspace Failure Case Logs.


* *Table III:* 10-Trial End-to-End Positioning Error & Success Rate across variable paper placements.


* *Table IV:* Classical vs. Deep Learning Emotion Recognition Accuracy (%), Confusion Matrix, and Inference Latency (ms/frame).


* *Figure Strip:* 6-frame sequence showing the robot executing each state of the demo.




7. **6. Discussion (Criterion C4 — 1.0 mark):** Honest analysis of limitations—specifically the myCobot 280's $280\text{ mm}$ workspace boundary, servo gear backlash ($\pm 1.5^\circ$ hysteresis under gravity), motion blur if camera frames are grabbed before the arm fully settles, and lighting sensitivity.


8. **7. Conclusion, References (IEEE style), and E-Signed Contribution Appendix** (explicitly noting AI tool usage as required by Page 5).



---

Whenever you are ready to begin coding, tell me which module you want to generate first—for example:

1. **`vision/aruco_board.py`** (to generate your printable A4 target sheet right now),
2. **`robot_bridge.py` + `main.py**` (the SSH TCP video/command server and state machine), or
3. **`kinematics/dh_model.py` + `fk_verification.py**` (the Spong DH implementation and error table generator).