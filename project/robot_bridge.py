"""
Robot Bridge Server for myCobot 280 Jetson Nano.

Runs on the Jetson Nano as a temporary RPC & Video Streaming Server.
Copied via SCP at start of lab slot and executed with:
    python3 robot_bridge.py --port /dev/ttyTHS1 --cam 0

Features:
- Video Streaming Server (Port 5000): Encodes camera frames as JPEG (Quality 80) and streams over TCP socket.
- Command RPC Server (Port 5001): Receives JSON commands from user laptop:
    - {"cmd": "send_angles", "angles": [...], "speed": 30}
    - {"cmd": "send_coords", "coords": [...], "speed": 30, "mode": 0}
    - {"cmd": "get_angles"} -> responds {"status": "ok", "angles": [...]}
    - {"cmd": "get_coords"} -> responds {"status": "ok", "coords": [...]}
    - {"cmd": "power_on"}
    - {"cmd": "power_off"}
    - {"cmd": "ping"} -> responds {"status": "pong"}
"""

import socket
import threading
import json
import struct
import time
import argparse
import sys

# Default ports
VIDEO_PORT = 5000
CMD_PORT = 5001


class JetsonRobotBridge:
    def __init__(self, serial_port: str = "/dev/ttyTHS1", baud: int = 1000000, cam_id: int = 0, is_mock: bool = False):
        self.serial_port = serial_port
        self.baud = baud
        self.cam_id = cam_id
        self.is_mock = is_mock
        self.running = True

        self.current_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.current_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.mc = None
        if not self.is_mock:
            try:
                from pymycobot.mycobot280 import MyCobot280
                print(f"Connecting to MyCobot on {serial_port} at {baud} baud...")
                self.mc = MyCobot280(serial_port, baud)
                self.mc.power_on()
                time.sleep(1.0)
                print("MyCobot initialized successfully.")
            except Exception as e:
                print(f"Hardware init warning: {e}. Switching to mock simulation.")
                self.is_mock = True

    def start(self, host: str = "0.0.0.0"):
        t_video = threading.Thread(target=self._run_video_server, args=(host, VIDEO_PORT), daemon=True)
        t_cmd = threading.Thread(target=self._run_cmd_server, args=(host, CMD_PORT), daemon=True)

        t_video.start()
        t_cmd.start()

        print(f"==================================================")
        print(f" Jetson Robot Bridge Server Active")
        print(f" - Video Streamer:  tcp://{host}:{VIDEO_PORT}")
        print(f" - Command Listener: tcp://{host}:{CMD_PORT}")
        print(f" Mode: {'MOCK SIMULATION' if self.is_mock else 'HARDWARE CONNECTED'}")
        print(f" Press Ctrl+C to shutdown")
        print(f"==================================================")

        try:
            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down robot bridge...")
            self.running = False

    def _run_cmd_server(self, host: str, port: int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)

        while self.running:
            try:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client_cmd, args=(conn, addr), daemon=True).start()
            except Exception as e:
                if self.running:
                    print(f"Cmd server accept error: {e}")

    def _handle_client_cmd(self, conn: socket.socket, addr):
        conn.settimeout(10.0)
        buffer = ""
        while self.running:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    res = self._execute_command(req)
                    conn.sendall((json.dumps(res) + "\n").encode("utf-8"))
            except socket.timeout:
                continue
            except Exception:
                break
        conn.close()

    def _execute_command(self, req: dict) -> dict:
        cmd = req.get("cmd", "")
        if cmd == "ping":
            return {"status": "pong", "time": time.time()}

        if self.is_mock or self.mc is None:
            # Mock behavior
            if cmd == "send_angles":
                self.current_angles = req.get("angles", self.current_angles)
                return {"status": "ok", "angles": self.current_angles}
            elif cmd == "send_coords":
                self.current_coords = req.get("coords", self.current_coords)
                return {"status": "ok", "coords": self.current_coords}
            elif cmd == "get_angles":
                return {"status": "ok", "angles": self.current_angles}
            elif cmd == "get_coords":
                return {"status": "ok", "coords": self.current_coords}
            elif cmd in ["power_on", "power_off", "release_all_servos"]:
                return {"status": "ok"}
            return {"status": "error", "message": f"Unknown mock cmd: {cmd}"}

        # Real Hardware behavior
        try:
            if cmd == "send_angles":
                angles = req.get("angles")
                speed = req.get("speed", 30)
                self.mc.send_angles(angles, speed)
                return {"status": "ok"}
            elif cmd == "send_coords":
                coords = req.get("coords")
                speed = req.get("speed", 30)
                mode = req.get("mode", 0)
                self.mc.send_coords(coords, speed, mode)
                return {"status": "ok"}
            elif cmd == "get_angles":
                angles = self.mc.get_angles()
                return {"status": "ok", "angles": angles}
            elif cmd == "get_coords":
                coords = self.mc.get_coords()
                return {"status": "ok", "coords": coords}
            elif cmd == "power_on":
                self.mc.power_on()
                return {"status": "ok"}
            elif cmd == "power_off":
                self.mc.power_off()
                return {"status": "ok"}
            elif cmd == "release_all_servos":
                self.mc.release_all_servos()
                return {"status": "ok"}
            else:
                return {"status": "error", "message": f"Unknown hardware cmd: {cmd}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _run_video_server(self, host: str, port: int):
        try:
            import cv2
        except ImportError:
            print("OpenCV not found. Video server running with dummy frames.")
            cv2 = None

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(2)

        cap = None
        if cv2 is not None and not self.is_mock:
            cap = cv2.VideoCapture(self.cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            try:
                conn, addr = s.accept()
                print(f"Video client connected from {addr}")
                while self.running:
                    if cap is not None and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            continue
                    else:
                        # Simulated mock video frame
                        import numpy as np
                        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                        if cv2 is not None:
                            cv2.putText(frame, "MOCK CAMERA STREAM - JETSON BRIDGE", (50, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if cv2 is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        res, encimg = cv2.imencode('.jpg', frame, encode_param)
                        if not res:
                            continue
                        data = encimg.tobytes()
                    else:
                        data = b"DUMMY_FRAME"

                    size = len(data)
                    conn.sendall(struct.pack(">L", size) + data)
                    time.sleep(0.033)  # ~30 FPS
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="myCobot 280 Jetson Nano Bridge Server")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Serial port for robot arm.")
    parser.add_argument("--baud", type=int, default=1000000, help="Baudrate for robot arm.")
    parser.add_argument("--cam", type=int, default=0, help="Flange camera device index.")
    parser.add_argument("--mock", action="store_true", help="Run with mock robot/camera simulation.")

    args = parser.parse_args()
    bridge = JetsonRobotBridge(serial_port=args.port, baud=args.baud, cam_id=args.cam, is_mock=args.mock)
    bridge.start()

