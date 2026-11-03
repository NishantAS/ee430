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


class RobotError(RuntimeError):
    pass


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def within_limits(q: Sequence[float]) -> bool:
    return all(lo <= a <= hi for a, (lo, hi) in zip(q, config.JOINT_LIMITS_DEG))


class _RobotBase:
    """Blocking, verified moves shared by the real and the mock robot."""

    # subclasses implement these five
    def send_angles(self, angles, speed=config.SPEED): raise NotImplementedError
    def send_coords(self, coords, speed=config.SPEED, mode=0): raise NotImplementedError
    def get_angles(self) -> List[float]: raise NotImplementedError
    def get_coords(self) -> List[float]: raise NotImplementedError
    def close(self): pass

    def _wait_settled(self, read, timeout, window: float = 0.6, still: float = 0.3, min_time: float = 1.0):
        """
        Polls read() until the arm has stopped: no reading in the last `window` seconds differs by more than
        `still` (mm or deg) from the newest one. Returns that reading; raises RobotError on timeout.
        min_time skips the instant after the command, before the servos start moving.
        """
        t0, history = time.time(), []
        while time.time() - t0 < timeout:
            try:
                history.append((time.time(), np.asarray(read(), float)))
            except RobotError:
                pass
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

    def move_coords(self, coords: Sequence[float], speed: int = config.SPEED, mode: int = 0,
                    timeout: float = config.MOVE_TIMEOUT_S) -> List[float]:
        """
        Task-space move: send_coords (the robot's built-in IK; mode 0 = joint-interpolated, 1 = straight line).
        Waits until the arm stops, then checks it is within POS_TOL_MM of the target. send_coords never reports
        failure: for an unreachable target the arm simply does not move (or stops short), which this detects.
        Returns the coords the robot reports where it stopped.
        """
        coords = [float(c) for c in coords]
        self.send_coords(coords, speed, mode)
        now = self._wait_settled(self.get_coords, timeout)
        err = float(np.linalg.norm(now[:3] - coords[:3]))
        if err > config.POS_TOL_MM:
            raise RobotError(f"Robot stopped {err:.1f} mm from {np.round(coords, 1).tolist()} "
                             f"(at {np.round(now, 1).tolist()}). Probably outside the reachable workspace "
                             "for this orientation.")
        return now.tolist()

    def move_tool(self, xyz: Sequence[float], orientation: Sequence[float], speed: int = config.SPEED,
                  mode: int = 0) -> List[float]:
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

    def send_coords(self, coords, speed=config.SPEED, mode=0):
        self._call("send_coords", coords=[float(c) for c in coords], speed=int(speed), mode=int(mode))

    def get_angles(self):
        return self._call("get_angles")["angles"]

    def get_coords(self):
        return self._call("get_coords")["coords"]

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
        from kinematics.dh_model import flange_coords
        self._fk = flange_coords
        self._q = list(config.HOME_ANGLES)
        self._coords = flange_coords(self._q)
        print("Using MOCK robot (no hardware).")

    def send_angles(self, angles, speed=config.SPEED):
        self._q = [float(a) for a in angles]
        self._coords = self._fk(self._q)

    def send_coords(self, coords, speed=config.SPEED, mode=0):
        self._coords = [float(c) for c in coords]

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
