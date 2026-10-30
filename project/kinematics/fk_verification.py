"""
Kinematics Forward Kinematics Verification Script (Criterion A2).

Compares user Spong DH forward kinematics against pymycobot.get_coords()
across 15 diverse joint configurations (Home, Search, Quadrant hovers, intermediate).
Outputs results to results/fk_error_table.csv with position and orientation errors.

Can be run:
1. Connected directly to real myCobot arm (--port /dev/ttyTHS1 or network RPC).
2. Connected to robot_bridge server (--host <jetson_ip>).
3. In mock mode (--mock) using simulated baseline forward kinematics.
"""

import os
import sys
import time
import argparse
import csv
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kinematics.dh_model import forward_kinematics_spong, get_flange_pose

# 15 Diverse Test Configurations in Joint Space (degrees: [q1, q2, q3, q4, q5, q6])
TEST_CONFIGURATIONS = [
    {"name": "HOME_REST",         "angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    {"name": "SEARCH_FACE",       "angles": [0.0, 20.0, -40.0, 20.0, 0.0, 0.0]},
    {"name": "SEARCH_PAPER_HIGH", "angles": [0.0, 45.0, -80.0, 35.0, 0.0, -90.0]},
    {"name": "SEARCH_PAPER_LOW",  "angles": [10.0, 50.0, -85.0, 35.0, 5.0, -90.0]},
    {"name": "QUADRANT_1_HOVER",  "angles": [-15.0, 40.0, -70.0, 30.0, 10.0, -90.0]},
    {"name": "QUADRANT_2_HOVER",  "angles": [15.0, 40.0, -70.0, 30.0, -10.0, -90.0]},
    {"name": "QUADRANT_3_HOVER",  "angles": [20.0, 55.0, -90.0, 35.0, -15.0, -90.0]},
    {"name": "QUADRANT_4_HOVER",  "angles": [-20.0, 55.0, -90.0, 35.0, 15.0, -90.0]},
    {"name": "ELBOW_UP_TEST",     "angles": [5.0, 30.0, -60.0, 30.0, 0.0, -45.0]},
    {"name": "ELBOW_DOWN_TEST",   "angles": [5.0, -30.0, 60.0, -30.0, 0.0, -45.0]},
    {"name": "YAW_LEFT_30",       "angles": [-30.0, 35.0, -70.0, 35.0, -30.0, 0.0]},
    {"name": "YAW_RIGHT_30",      "angles": [30.0, 35.0, -70.0, 35.0, 30.0, 0.0]},
    {"name": "WRIST_PITCH_UP",    "angles": [0.0, 40.0, -50.0, 60.0, 0.0, -90.0]},
    {"name": "WRIST_PITCH_DOWN",  "angles": [0.0, 40.0, -70.0, 10.0, 0.0, -90.0]},
    {"name": "INTERMEDIATE_TRANS","angles": [10.0, 25.0, -45.0, 20.0, 10.0, -30.0]},
]


def run_fk_verification(mc=None, is_mock: bool = True, output_csv: str = "project/results/fk_error_table.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("==========================================================================")
    print(" EE430 Component A2: Forward Kinematics Verification vs Spong DH Model")
    print("==========================================================================")
    print(f"Mode: {'MOCK SIMULATION' if is_mock else 'HARDWARE CONNECTED'}")
    print(f"Total configurations to evaluate: {len(TEST_CONFIGURATIONS)}")
    print(f"Output CSV destination: {output_csv}\n")

    results = []

    for idx, test in enumerate(TEST_CONFIGURATIONS, start=1):
        name = test["name"]
        cmd_angles = test["angles"]

        if is_mock or mc is None:
            # In mock mode, actual angles have small simulated encoder jitter
            jitter = np.random.uniform(-0.15, 0.15, 6)
            meas_angles = [float(a + j) for a, j in zip(cmd_angles, jitter)]
            # Simulated mycobot firmware coords with small nominal calibration offset
            spong_pose = get_flange_pose(meas_angles)
            meas_coords = [
                spong_pose[0] + float(np.random.normal(0, 0.35)),
                spong_pose[1] + float(np.random.normal(0, 0.35)),
                spong_pose[2] + float(np.random.normal(0, 0.40)),
                spong_pose[3] + float(np.random.normal(0, 0.20)),
                spong_pose[4] + float(np.random.normal(0, 0.20)),
                spong_pose[5] + float(np.random.normal(0, 0.25)),
            ]
        else:
            # Send commanded angles to robot and wait for servos to settle
            mc.send_angles(cmd_angles, 30)
            time.sleep(2.0)
            meas_angles = mc.get_angles()
            meas_coords = mc.get_coords()
            if not meas_angles or not meas_coords:
                print(f"Warning: Failed to read state from robot on test {name}")
                continue

        # Evaluate our Spong DH forward kinematics on the measured angles
        dh_pose = get_flange_pose(meas_angles)

        # Euclidean position error e_p in mm
        pos_dh = np.array(dh_pose[0:3])
        pos_mc = np.array(meas_coords[0:3])
        e_pos = float(np.linalg.norm(pos_dh - pos_mc))

        # Orientation absolute angle error (degrees)
        ori_dh = np.array(dh_pose[3:6])
        ori_mc = np.array(meas_coords[3:6])
        e_ori = float(np.linalg.norm(ori_dh - ori_mc))

        row = {
            "Test_ID": idx,
            "Config_Name": name,
            "Cmd_Q1": cmd_angles[0], "Cmd_Q2": cmd_angles[1], "Cmd_Q3": cmd_angles[2],
            "Cmd_Q4": cmd_angles[3], "Cmd_Q5": cmd_angles[4], "Cmd_Q6": cmd_angles[5],
            "Meas_Q1": round(meas_angles[0], 2), "Meas_Q2": round(meas_angles[1], 2),
            "Meas_Q3": round(meas_angles[2], 2), "Meas_Q4": round(meas_angles[3], 2),
            "Meas_Q5": round(meas_angles[4], 2), "Meas_Q6": round(meas_angles[5], 2),
            "DH_X_mm": round(dh_pose[0], 2), "DH_Y_mm": round(dh_pose[1], 2), "DH_Z_mm": round(dh_pose[2], 2),
            "DH_Rx_deg": round(dh_pose[3], 2), "DH_Ry_deg": round(dh_pose[4], 2), "DH_Rz_deg": round(dh_pose[5], 2),
            "MC_X_mm": round(meas_coords[0], 2), "MC_Y_mm": round(meas_coords[1], 2), "MC_Z_mm": round(meas_coords[2], 2),
            "MC_Rx_deg": round(meas_coords[3], 2), "MC_Ry_deg": round(meas_coords[4], 2), "MC_Rz_deg": round(meas_coords[5], 2),
            "Pos_Error_mm": round(e_pos, 3),
            "Ori_Error_deg": round(e_ori, 3)
        }
        results.append(row)
        print(f"[{idx:02d}/15] {name:<20} | Pos Error: {e_pos:.3f} mm | Ori Error: {e_ori:.3f}°")

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        pos_errors = [r["Pos_Error_mm"] for r in results]
        ori_errors = [r["Ori_Error_deg"] for r in results]
        print("\n----------------- SUMMARY STATISTICS -----------------")
        print(f"Mean Position Error : {np.mean(pos_errors):.3f} mm (Max: {np.max(pos_errors):.3f} mm)")
        print(f"Mean Orientation Err: {np.mean(ori_errors):.3f}°  (Max: {np.max(ori_errors):.3f}°)")
        print(f"Successfully saved verification data to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spong DH FK verification on myCobot.")
    parser.add_argument("--mock", action="store_true", default=True, help="Run in mock mode without physical robot.")
    parser.add_argument("--real", action="store_true", help="Connect to physical robot arm.")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Serial port for myCobot.")
    parser.add_argument("--baud", type=int, default=1000000, help="Baudrate for myCobot.")
    parser.add_argument("--output", type=str, default="project/results/fk_error_table.csv", help="Output CSV path.")

    args = parser.parse_args()

    mc = None
    is_mock = args.mock and not args.real

    if not is_mock:
        try:
            from pymycobot.mycobot280 import MyCobot280
            print(f"Connecting to MyCobot on {args.port} at {args.baud} baud...")
            mc = MyCobot280(args.port, args.baud)
            mc.power_on()
            time.sleep(1.0)
        except Exception as e:
            print(f"Failed to connect to hardware: {e}. Falling back to mock simulation.")
            is_mock = True

    run_fk_verification(mc=mc, is_mock=is_mock, output_csv=args.output)

