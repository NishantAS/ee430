"""
Quick Utility Script: Reset myCobot to Zero Pose [0, 0, 0, 0, 0, 0].

Can be executed from your laptop over the network:
    python project/reset_home.py --host 10.42.0.1

Or run directly on the Jetson Nano:
    python project/reset_home.py --direct --port /dev/ttyTHS1
"""

import argparse
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Calibrated Home / Rest pose matching physical camera alignment
# End-effector pose [x, y, z (mm), rx, ry, rz (deg)] for the level-camera home configuration
BASE_POSE = [51.95, -64.45, 410.32, -91.56, 52.47, -91.07]


def reset_to_zero(host: str = "10.42.0.1", direct: bool = False, port: str = "/dev/ttyTHS1", speed: int = 25):
    print("==================================================")
    print(" myCobot 280: Base Pose Reset")
    print(f" Target Coords: {BASE_POSE}")
    print("==================================================")

    if direct:
        try:
            from pymycobot.mycobot280 import MyCobot280
            print(f"Connecting directly via serial on {port} (1000000 baud)...")
            mc = MyCobot280(port, 1000000)
            mc.power_on()
            time.sleep(0.5)
            print(f"Moving arm to base pose at speed={speed}...")
            mc.send_coords(BASE_POSE, speed, 0)
            time.sleep(3.0)
            curr = mc.get_coords()
            print(f"Current Coords: {curr}")
            print("Done! Robot is in base position.")
        except Exception as e:
            print(f"Direct connection error: {e}")
    else:
        from project.robot_client import RobotRemoteClient
        print(f"Connecting to Jetson Robot Bridge at {host}:5001...")
        client = RobotRemoteClient(host=host, mock=False)
        if client.cmd_sock is None:
            print("Error: Could not connect to robot bridge. Ensure robot_bridge.py is running on Jetson.")
            return

        print(f"Sending Base Pose command {BASE_POSE} at speed={speed}...")
        client.send_coords(BASE_POSE, speed=speed, mode=0)
        time.sleep(3.0)
        curr = client.get_coords()
        print(f"Current Coords: {curr}")
        print("Done! Robot is in base position.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset myCobot to zero position.")
    parser.add_argument("--host", type=str, default="10.42.0.1", help="Jetson Nano IP address.")
    parser.add_argument("--direct", action="store_true", help="Connect directly via serial (on Jetson).")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Serial port for direct connection.")
    parser.add_argument("--speed", type=int, default=25, help="Movement speed (1-100).")

    args = parser.parse_args()
    reset_to_zero(host=args.host, direct=args.direct, port=args.port, speed=args.speed)

