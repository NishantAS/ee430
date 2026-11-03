"""
Small robot utilities (connect to robot_bridge.py on the Jetson first).

    python robot_tools.py status                 # joint angles, get_coords, our DH FK and their difference
    python robot_tools.py home                   # all joints to 0 (arm straight up) - do this before leaving
    python robot_tools.py overview               # go to the calibrated overview pose
    python robot_tools.py goto X Y Z [RX RY RZ]  # send_coords; orientation defaults to the overview one
    python robot_tools.py release                # servos limp (HOLD THE ARM), e.g. to move it by hand
    python robot_tools.py lock                   # servos hold again

Options: --host <jetson ip>, --mock, --speed N, --linear (goto in a straight line)
"""

import argparse

import numpy as np

import config
from common import calibration_store as store
from common.robot_client import connect
from kinematics import dh_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "home", "overview", "goto", "release", "lock"])
    ap.add_argument("coords", nargs="*", type=float, help="goto target: X Y Z [RX RY RZ] (mm, deg)")
    ap.add_argument("--host", default=config.DEFAULT_HOST)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--speed", type=int, default=config.SPEED)
    ap.add_argument("--linear", action="store_true", help="goto: straight-line motion (send_coords mode 1)")
    args = ap.parse_args()
    robot = connect(args.host, args.mock)
    fmt = lambda v: np.round(v, 2).tolist()

    if args.action == "status":
        q = robot.get_angles()
        firmware, ours = np.array(robot.get_coords()), np.array(dh_model.flange_coords(q))
        print(f"angles (deg)        : {fmt(q)}")
        print(f"get_coords          : {fmt(firmware)}")
        print(f"our DH FK           : {fmt(ours)}")
        print(f"position difference : {np.linalg.norm(firmware[:3] - ours[:3]):.2f} mm")
    elif args.action == "home":
        robot.move_angles(config.HOME_ANGLES, args.speed)
    elif args.action == "overview":
        robot.move_angles(store.load(require=["overview_angles"])["overview_angles"], args.speed)
    elif args.action == "goto":
        if len(args.coords) == 3:
            orient = store.load().get("overview_coords", config.DEFAULT_OVERVIEW_COORDS)[3:]
            target = args.coords + list(orient)
        elif len(args.coords) == 6:
            target = args.coords
        else:
            ap.error("goto needs X Y Z or X Y Z RX RY RZ")
        print(f"reached {fmt(robot.move_coords(target, args.speed, 1 if args.linear else 0))}")
    elif args.action == "release":
        input("HOLD THE ARM - it will go limp. Press ENTER to release ...")
        robot.release()
    elif args.action == "lock":
        robot.lock()
    robot.close()


if __name__ == "__main__":
    main()
