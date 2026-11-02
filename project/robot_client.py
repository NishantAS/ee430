"""
Robot Remote RPC Client & Video Receiver.

Connects to robot_bridge.py running on Jetson Nano over TCP for motion commands only (port 5001:
send_angles, send_coords, get_angles, get_coords). The flange camera is plugged into the laptop and is read
locally with OpenCV (background thread, newest frame only).

Supports fallback to local mock mode if --mock is passed.
"""

import socket
import json
import struct
import time
import threading
import numpy as np
import cv2
from typing import Optional, List, Tuple


class RobotRemoteClient:
    def __init__(self, host: str = "127.0.0.1", cmd_port: int = 5001, mock: bool = False, camera_index: Optional[int] = None):
        self.host = host
        self.camera_index = camera_index
        self.cmd_port = cmd_port
        self.mock = mock

        self.cmd_sock = None
        self.mock_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mock_coords = [150.0, 0.0, 200.0, 180.0, 0.0, -90.0]

        # Newest decoded frame, filled by a background thread so slow vision code never backs up the socket
        self._latest_frame = None
        self._frame_id = 0
        self._frame_cond = threading.Condition()
        self._last_read_id = -1
        self._stream_running = False
        self._cap = None

        if not self.mock:
            self._connect()
        if camera_index is not None:
            self._open_camera()

    def _connect(self):
        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_sock.connect((self.host, self.cmd_port))
            print(f"Connected to Robot Command Server at {self.host}:{self.cmd_port}")
        except Exception as e:
            print(f"Warning: Could not connect to command server ({e}). Falling back to mock robot.")
            self.mock = True

    def _open_camera(self):
        """Opens the laptop-attached camera (MJPG, tiny buffer) and starts the newest-frame reader thread."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Warning: Cannot open camera index {self.camera_index}. Using simulated frames.")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._stream_running = True
        threading.Thread(target=self._camera_reader, daemon=True).start()
        print(f"Camera {self.camera_index} opened ({int(cap.get(3))}x{int(cap.get(4))})")

    def send_angles(self, angles: List[float], speed: int = 30) -> bool:
        if self.mock or self.cmd_sock is None:
            self.mock_angles = list(angles)
            return True
        req = {"cmd": "send_angles", "angles": angles, "speed": speed}
        res = self._send_cmd(req)
        return res.get("status") == "ok"

    def send_coords(self, coords: List[float], speed: int = 30, mode: int = 0) -> bool:
        if self.mock or self.cmd_sock is None:
            self.mock_coords = list(coords)
            return True
        req = {"cmd": "send_coords", "coords": coords, "speed": speed, "mode": mode}
        res = self._send_cmd(req)
        return res.get("status") == "ok"

    def get_angles(self) -> List[float]:
        if self.mock or self.cmd_sock is None:
            jitter = np.random.uniform(-0.1, 0.1, 6)
            return [float(a + j) for a, j in zip(self.mock_angles, jitter)]
        req = {"cmd": "get_angles"}
        res = self._send_cmd(req)
        return res.get("angles", self.mock_angles)

    def get_coords(self) -> List[float]:
        if self.mock or self.cmd_sock is None:
            jitter = np.random.uniform(-0.2, 0.2, 6)
            return [float(c + j) for c, j in zip(self.mock_coords, jitter)]
        req = {"cmd": "get_coords"}
        res = self._send_cmd(req)
        return res.get("coords", self.mock_coords)

    def _send_cmd(self, req: dict) -> dict:
        try:
            msg = json.dumps(req) + "\n"
            self.cmd_sock.sendall(msg.encode("utf-8"))
            data = self.cmd_sock.recv(4096).decode("utf-8")
            return json.loads(data)
        except Exception as e:
            print(f"RPC communication error: {e}")
            return {"status": "error", "message": str(e)}

    def _camera_reader(self):
        """Grabs frames as fast as the camera delivers them; only the newest one is kept."""
        while self._stream_running:
            ret, img = self._cap.read()
            if not ret or img is None:
                time.sleep(0.01)
                continue
            with self._frame_cond:
                self._latest_frame = img
                self._frame_id += 1
                self._frame_cond.notify_all()

    def read_frame(self, local_cap=None, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Returns the newest frame: waits (up to timeout) for one that has not been returned yet,
        so callers always get a fresh image and never a queued-up stale one.
        """
        if self._stream_running:
            with self._frame_cond:
                self._frame_cond.wait_for(lambda: self._frame_id != self._last_read_id, timeout=timeout)
                if self._latest_frame is not None:
                    self._last_read_id = self._frame_id
                    return self._latest_frame.copy()

        # Synthetic fallback frame
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(frame, "SIMULATED TEST VIDEO FEED", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
