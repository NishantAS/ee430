"""
Runs ON THE JETSON NANO. Thin JSON-over-TCP wrapper around pymycobot so all logic stays on the laptop.

    scp robot_bridge.py nano@10.42.0.1:/tmp/          (copy at the start of the slot)
    ssh nano@10.42.0.1 python3 /tmp/robot_bridge.py   (Ctrl+C to stop; delete /tmp/robot_bridge.py when done)

Protocol: one JSON object per line. Request {"cmd": ..., ...}, reply {"status": "ok"|"pong"|"error", ...}.
Commands: ping, send_angles, send_coords, get_angles, get_coords, stop, release_all_servos, lock_here, power_on, power_off.
Kept Python 3.6 compatible (Jetson Nano image) and dependency-free apart from pymycobot.
"""

import argparse
import json
import socket
import threading
import time


def open_robot(port, baud):
    try:
        from pymycobot.mycobot280 import MyCobot280 as Robot
    except ImportError:  # older pymycobot on the Jetson image
        from pymycobot.mycobot import MyCobot as Robot
    mc = Robot(port, baud)
    time.sleep(0.5)
    mc.power_on()
    time.sleep(0.5)
    return mc


class Bridge:
    def __init__(self, mc):
        self.mc = mc
        self.lock = threading.Lock()  # one serial transaction at a time

    def _read(self, fn, retries=5):
        """get_angles/get_coords sometimes return -1 or [] (serial hiccup): retry a few times."""
        for _ in range(retries):
            val = fn()
            if isinstance(val, list) and len(val) == 6:
                return val
            time.sleep(0.05)
        raise RuntimeError("robot returned no valid reading (%r)" % (val,))

    def execute(self, req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"status": "pong"}
        with self.lock:
            if cmd == "send_angles":
                self.mc.send_angles([float(a) for a in req["angles"]], int(req.get("speed", 30)))
                return {"status": "ok"}
            if cmd == "send_coords":  # the robot's own IK; mode 0 = joint-interpolated, 1 = linear
                self.mc.send_coords([float(c) for c in req["coords"]], int(req.get("speed", 30)),
                                    int(req.get("mode", 0)))
                return {"status": "ok"}
            if cmd == "get_angles":
                return {"status": "ok", "angles": self._read(self.mc.get_angles)}
            if cmd == "get_coords":
                return {"status": "ok", "coords": self._read(self.mc.get_coords)}
            if cmd in ("release_all_servos", "power_on", "power_off", "stop"):
                getattr(self.mc, cmd)()
                return {"status": "ok"}
            if cmd == "lock_here":
                return {"status": "ok", "angles": self._lock_here()}
        return {"status": "error", "message": "unknown command %r" % cmd}

    def _lock_here(self):
        """
        After release_all_servos: stiffen the arm WHERE IT IS NOW. Reads the angles it is being held at,
        powers the servos and immediately commands those same angles, so it cannot jump back to the last
        commanded pose. Returns the angles it locked at.
        """
        angles = self._read(self.mc.get_angles)
        if hasattr(self.mc, "focus_all_servos"):
            self.mc.focus_all_servos()
        elif hasattr(self.mc, "focus_servo"):  # older pymycobot
            for i in range(1, 7):
                self.mc.focus_servo(i)
        else:
            self.mc.power_on()
        self.mc.send_angles(angles, 20)
        return angles

    def handle(self, conn, addr):
        print("client connected: %s:%d" % addr)
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        try:
            for line in f:
                if not line.strip():
                    continue
                try:
                    res = self.execute(json.loads(line))
                except Exception as e:  # report every failure to the laptop instead of dying
                    res = {"status": "error", "message": str(e)}
                f.write(json.dumps(res) + "\n")
                f.flush()
        except OSError:
            pass
        finally:
            conn.close()
            print("client disconnected: %s:%d" % addr)


def main():
    ap = argparse.ArgumentParser(description="myCobot 280 Jetson Nano bridge")
    ap.add_argument("--port", default="/dev/ttyTHS1", help="robot serial port")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--listen", type=int, default=5001, help="TCP port")
    args = ap.parse_args()

    bridge = Bridge(open_robot(args.port, args.baud))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.listen))
    srv.listen(2)
    print("robot_bridge listening on tcp/%d (Ctrl+C to stop)" % args.listen)
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=bridge.handle, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
