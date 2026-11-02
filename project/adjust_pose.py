"""
Interactive end-effector adjustment utility for myCobot 280 (works in coordinates, not joint angles).

Examples:
    # Show the current end-effector pose:
    python project/adjust_pose.py --host 10.42.0.1

    # Set one component (mm / deg), e.g. spin the camera about its axis by changing rz:
    python project/adjust_pose.py --host 10.42.0.1 --axis rz --value -120

    # Nudge a component relative to where it is now:
    python project/adjust_pose.py --host 10.42.0.1 --axis z --delta 10
"""

import argparse
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from project.robot_client import RobotRemoteClient

AXES = ["x", "y", "z", "rx", "ry", "rz"]


def adjust(host="10.42.0.1", axis=None, value=None, delta=None, speed=20, mode=0):
    client = RobotRemoteClient(host=host, mock=False)
    if client.cmd_sock is None:
        print("Error: Could not connect to robot bridge. Ensure robot_bridge.py is running on Jetson.")
        return

    pose = client.get_coords()
    print("--------------------------------------------------")
    for name, v in zip(AXES, pose):
        print(f"  {name:>2}: {v:8.2f}")
    print("--------------------------------------------------")

    if axis is None or (value is None and delta is None):
        print("No adjustment specified. Use --axis [x|y|z|rx|ry|rz] with --value or --delta.")
        return

    new_pose = list(pose)
    i = AXES.index(axis)
    new_pose[i] = float(value) if value is not None else pose[i] + float(delta)
    print(f"Moving {axis}: {pose[i]:.2f} -> {new_pose[i]:.2f}")

    client.send_coords(new_pose, speed=speed, mode=mode)
    time.sleep(3.0)
    print(f"Updated pose: {[round(v, 2) for v in client.get_coords()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust the myCobot end-effector pose.")
    parser.add_argument("--host", type=str, default="10.42.0.1", help="Jetson Nano IP address.")
    parser.add_argument("--axis", type=str, choices=AXES, default=None)
    parser.add_argument("--value", type=float, default=None, help="Absolute target for the axis.")
    parser.add_argument("--delta", type=float, default=None, help="Relative change for the axis.")
    parser.add_argument("--speed", type=int, default=20, help="Movement speed (1-100).")
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0 = joint-interpolated, 1 = linear.")
    args = parser.parse_args()
    adjust(args.host, args.axis, args.value, args.delta, args.speed, args.mode)
