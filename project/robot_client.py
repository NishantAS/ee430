"""
Robot Remote RPC Client & Video Receiver.

Connects to robot_bridge.py running on Jetson Nano over TCP:
- Video on port 5000 (receives JPEG frames)
- Commands on port 5001 (sends send_angles, send_coords, get_angles, get_coords)

Supports fallback to local mock mode if --mock is passed.
"""

import socket
import json
import struct
import time
import numpy as np
import cv2
from typing import Optional, List, Tuple


class RobotRemoteClient:
    def __init__(self, host: str = "127.0.0.1", video_port: int = 5000, cmd_port: int = 5001, mock: bool = False):
        self.host = host
        self.video_port = video_port
        self.cmd_port = cmd_port
        self.mock = mock

        self.cmd_sock = None
        self.video_sock = None
        self.mock_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mock_coords = [150.0, 0.0, 200.0, 180.0, 0.0, -90.0]

        if not self.mock:
            self._connect()

    def _connect(self):
        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_sock.connect((self.host, self.cmd_port))
            print(f"Connected to Robot Command Server at {self.host}:{self.cmd_port}")
        except Exception as e:
            print(f"Warning: Could not connect to command server ({e}). Falling back to mock robot.")
            self.mock = True

        try:
            self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_sock.connect((self.host, self.video_port))
            print(f"Connected to Robot Video Server at {self.host}:{self.video_port}")
        except Exception as e:
            print(f"Warning: Could not connect to video server ({e}). Falling back to local camera/mock frames.")

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

    def read_frame(self, local_cap=None) -> Optional[np.ndarray]:
        """Reads frame either from TCP video stream, local webcam, or generates mock frame."""
        if self.video_sock is not None:
            try:
                raw_size = self._recv_exact(self.video_sock, 4)
                if not raw_size:
                    return None
                size = struct.unpack(">L", raw_size)[0]
                frame_data = self._recv_exact(self.video_sock, size)
                if not frame_data:
                    return None
                img = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img
            except Exception:
                pass

        if local_cap is not None and local_cap.isOpened():
            ret, frame = local_cap.read()
            if ret:
                return frame

        # Synthetic fallback frame
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(frame, "SIMULATED TEST VIDEO FEED", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def _recv_exact(self, sock: socket.socket, n_bytes: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n_bytes:
            packet = sock.recv(n_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

