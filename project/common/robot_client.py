"""
Laptop-side connection to robot_bridge.py (running on the Jetson Nano), which calls pymycobot.

Protocol: one JSON object per line in each direction. Every failure raises RobotError; nothing is ever
silently replaced by made-up values. Because send_coords does not report whether the robot's IK found a
solution, every move is verified by reading the robot back (get_angles / get_coords).
MockRobot has the same interface for dry runs without hardware.
"""

import json
import socket
import time
from typing import List, Sequence

import numpy as np

import config
from kinematics import dh_model


class RobotError(RuntimeError):
    pass


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def within_limits(q: Sequence[float]) -> bool:
    return all(lo <= a <= hi for a, (lo, hi) in zip(q, config.JOINT_LIMITS_DEG))


def elbow_clearly_up(q: Sequence[float]) -> bool:
    """Elbow bent up enough that the robot's IK will keep the inverted-V shape."""
    return q[2] <= -config.ELBOW_UP_MIN_BEND_DEG


def elbow_flipped(q: Sequence[float]) -> bool:
    """True if the arm is in (or flipping into) the elbow-down "V" shape. In our DH convention q3 < 0 is
    elbow-up (inverted V, clear of the base) and q3 > 0 is elbow-down (checked with the FK geometry)."""
    return q[2] > config.ELBOW_FLIP_Q3_DEG


class _RobotBase:
    """Blocking, verified moves shared by the real and the mock robot."""

    # subclasses implement these
    def send_angles(self, angles, speed=config.SPEED): raise NotImplementedError
    def send_coords(self, coords, speed=config.SPEED, mode=1): raise NotImplementedError
    def get_angles(self) -> List[float]: raise NotImplementedError
    def get_coords(self) -> List[float]: raise NotImplementedError
    def stop(self): raise NotImplementedError
    def close(self): pass

    def _wait_settled(self, read, timeout, check=None, window: float = 0.6, still: float = 0.3,
                      min_time: float = 1.0):
        """
        Polls read() until the arm has stopped: no reading in the last `window` seconds differs by more than
        `still` (mm or deg) from the newest one. Returns that reading; raises RobotError on timeout.
        check(reading) runs on every reading and may raise to abort the move.
        min_time skips the instant after the command, before the servos start moving.
        """
        t0, history = time.time(), []
        while time.time() - t0 < timeout:
            try:
                reading = np.asarray(read(), float)
            except RobotError:
                time.sleep(0.1)
                continue
            if check is not None:
                check(reading)
            history.append((time.time(), reading))
            now = time.time()
            recent = [v for t, v in history if now - t <= window]
            if (now - t0 >= min_time and len(recent) >= 3 and history[0][0] <= now - window
                    and max(np.max(np.abs(v - recent[-1])) for v in recent) < still):
                return recent[-1]
            time.sleep(0.1)
        last = np.round(history[-1][1], 1).tolist() if history else "no reading"
        raise RobotError(f"Arm did not come to rest within {timeout:.0f} s (last reading {last})")

    def move_angles(self, angles: Sequence[float], speed: int = config.SPEED,
                    timeout: float = config.MOVE_TIMEOUT_S) -> None:
        """Joint-space move (send_angles): waits until the arm stops, then checks every joint is within
        ANGLE_TOL_DEG of the command."""
        angles = [float(a) for a in angles]
        if not within_limits(angles):
            raise RobotError(f"Refusing angles outside joint limits: {np.round(angles, 1).tolist()}")
        self.send_angles(angles, speed)
        now = self._wait_settled(self.get_angles, timeout)
        err = float(np.max(np.abs(wrap180(now - angles))))
        if err > config.ANGLE_TOL_DEG:
            raise RobotError(f"Arm stopped {err:.1f} deg away from angles {np.round(angles, 1).tolist()} "
                             f"(at {np.round(now, 1).tolist()})")

    def move_coords(self, coords: Sequence[float], speed: int = config.SPEED, mode: int = 1,
                    timeout: float = config.MOVE_TIMEOUT_S, guard_elbow: bool = True) -> List[float]:
        """
        Task-space move with the robot's built-in IK: send_coords (mode 1 = straight line, the default: the
        IK then follows the path continuously and stays in the current arm shape; mode 0 = joint-interpolated).

        While moving, the joint angles are watched: if the elbow flips into the "V" shape (which can hit the
        base) the arm is stopped at once. After it stops, position (POS_TOL_MM) and orientation (ORI_TOL_DEG)
        are checked: send_coords never reports failure, an unreachable target just is not reached.
        Returns the coords the robot reports where it stopped.
        """
        coords = [float(c) for c in coords]
        q_now = self.get_angles()
        if guard_elbow and not elbow_clearly_up(q_now):
            raise RobotError(f"Elbow is not clearly bent up (q3 = {q_now[2]:.0f} deg, need <= "
                             f"{-config.ELBOW_UP_MIN_BEND_DEG:.0f}): the robot's IK could fold the arm into the "
                             "V shape. Move to an elbow-up pose first: python robot_tools.py ready")

        def check(reading):  # reading = coords (6) + angles (6)
            if guard_elbow and elbow_flipped(reading[6:]):
                self.stop()
                raise RobotError(f"Stopped: the robot's IK was flipping the elbow into the V shape "
                                 f"(q3 = {reading[8]:.0f} deg) on the way to {np.round(coords, 1).tolist()}.")

        self.send_coords(coords, speed, mode)
        now = self._wait_settled(lambda: list(self.get_coords()) + list(self.get_angles()), timeout, check)[:6]
        pos_err = float(np.linalg.norm(now[:3] - coords[:3]))
        ori_err = dh_model.rotation_error_deg(dh_model.euler_zyx_to_matrix(*now[3:]),
                                              dh_model.euler_zyx_to_matrix(*coords[3:]))
        if pos_err > config.POS_TOL_MM or ori_err > config.ORI_TOL_DEG:
            raise RobotError(f"Robot stopped {pos_err:.1f} mm / {ori_err:.1f} deg from "
                             f"{np.round(coords, 1).tolist()} (at {np.round(now, 1).tolist()}). "
                             "Probably outside the reachable workspace for this orientation.")
        return now.tolist()

    def move_tool(self, xyz: Sequence[float], orientation: Sequence[float], speed: int = config.SPEED,
                  mode: int = 1) -> List[float]:
        """Tool tip to xyz (pointing straight down): flange = tip + (0, 0, TOOL_LENGTH_MM)."""
        x, y, z = xyz
        return self.move_coords([x, y, z + config.TOOL_LENGTH_MM, *orientation], speed, mode)


def tool_xyz(coords: Sequence[float]) -> np.ndarray:
    """Tool tip position from flange coords (tool pointing straight down)."""
    return np.array(coords[:3], float) - [0.0, 0.0, config.TOOL_LENGTH_MM]


class RobotClient(_RobotBase):
    def __init__(self, host: str = config.DEFAULT_HOST, port: int = config.CMD_PORT, timeout: float = 5.0):
        try:
            self._sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as e:
            raise RobotError(f"Cannot reach robot_bridge at {host}:{port} ({e}). "
                             "Is robot_bridge.py running on the Jetson?") from e
        self._file = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        if self._call("ping").get("status") != "pong":
            raise RobotError("Bridge did not answer ping")

    def _call(self, cmd: str, **kw) -> dict:
        try:
            self._file.write(json.dumps({"cmd": cmd, **kw}) + "\n")
            self._file.flush()
            line = self._file.readline()
        except OSError as e:
            raise RobotError(f"Connection lost during '{cmd}': {e}") from e
        if not line:
            raise RobotError(f"Bridge closed the connection during '{cmd}'")
        res = json.loads(line)
        if res.get("status") not in ("ok", "pong"):
            raise RobotError(f"'{cmd}' failed on the robot: {res.get('message')}")
        return res

    def send_angles(self, angles, speed=config.SPEED):
        self._call("send_angles", angles=[float(a) for a in angles], speed=int(speed))

    def send_coords(self, coords, speed=config.SPEED, mode=1):
        self._call("send_coords", coords=[float(c) for c in coords], speed=int(speed), mode=int(mode))

    def get_angles(self):
        return self._call("get_angles")["angles"]

    def get_coords(self):
        return self._call("get_coords")["coords"]

    def stop(self):
        self._call("stop")

    def release(self):
        """Servos go limp so the arm can be moved by hand. HOLD THE ARM before calling."""
        self._call("release_all_servos")

    def lock(self) -> List[float]:
        """Servos stiffen at the pose the arm is held in now (no jump). Returns the locked joint angles."""
        return self._call("lock_here")["angles"]

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


class MockRobot(_RobotBase):
    """Instant-moving stand-in for testing code paths without the arm (reaches every target)."""

    def __init__(self):
        self._q = list(config.HOME_ANGLES)
        self._coords = dh_model.flange_coords(self._q)
        print("Using MOCK robot (no hardware).")

    def send_angles(self, angles, speed=config.SPEED):
        self._q = [float(a) for a in angles]
        self._coords = dh_model.flange_coords(self._q)

    def send_coords(self, coords, speed=config.SPEED, mode=1):
        self._coords = [float(c) for c in coords]

    def stop(self):
        pass

    def get_angles(self):
        return [q + float(np.random.uniform(-0.05, 0.05)) for q in self._q]

    def get_coords(self):
        return [c + float(np.random.uniform(-0.1, 0.1)) for c in self._coords]

    def release(self):
        pass

    def lock(self):
        return self.get_angles()


def connect(host: str = config.DEFAULT_HOST, mock: bool = False) -> _RobotBase:
    return MockRobot() if mock else RobotClient(host)
