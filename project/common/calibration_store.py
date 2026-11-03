"""
Load/save calibration/calibration.json, the only file calibrate.py writes and main.py reads.

{
  "camera_index": 2,
  "overview_angles": [q1..q6],                  # top-down pose from which the camera sees the table
  "overview_coords": [x, y, z, rx, ry, rz],     # the robot's get_coords there; rx, ry, rz = hover orientation
  "table": {
     "pixel_points":  [[u, v], ...],            # marker centres seen from the overview pose
     "robot_points":  [[x, y, z], ...],         # same points touched with the tool tip (get_coords, mm)
     "H_pixel_to_base": 3x3,                   # homography: image pixel -> table (x, y) in base frame
     "table_z_mm": float,
     "check_error_mm": float | null            # error at an extra touched point not used for the fit
  }
}
"""

import json
from typing import Any, Dict

import config


def load(require=()) -> Dict[str, Any]:
    """Returns the stored calibration; exits with instructions if any key in `require` is missing."""
    data = json.loads(config.CALIB_FILE.read_text()) if config.CALIB_FILE.exists() else {}
    missing = [k for k in require if k not in data]
    if missing:
        steps = {"camera_index": "camera", "overview_angles": "overview", "overview_coords": "overview",
                 "table": "table"}
        raise SystemExit(f"Calibration missing: {', '.join(missing)}.\nRun: "
                         + " && ".join(f"python calibrate.py {s}" for s in dict.fromkeys(steps[k] for k in steps if k in missing)))
    return data


def update(**values) -> None:
    data = load()
    data.update(values)
    config.CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CALIB_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved {', '.join(values)} to {config.CALIB_FILE}")
