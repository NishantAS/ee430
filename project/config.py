"""
Single place for every tunable constant, path and default used by the project.
Change values here, not inside the modules.
"""

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
CALIB_FILE = ROOT / "calibration" / "calibration.json"   # written by calibrate.py, read by everything else
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
DATASET_DIR = ROOT / "dl" / "dataset"                     # gesture images from experiments.py collect
SHEET_IMAGE = ASSETS_DIR / "target_sheet_A4.png"
HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")

# --------------------------------------------------------------------------- network
DEFAULT_HOST = "10.42.0.1"   # Jetson Nano address when sharing the laptop's wired connection
CMD_PORT = 5001

# --------------------------------------------------------------------------- robot motion
# All motion uses pymycobot's built-in kinematics: send_angles (joint space) and send_coords (task space,
# the robot solves the IK). Coordinates are the flange centre [x, y, z (mm), rx, ry, rz (deg)] in the base frame.
# myCobot 280 joint limits (vendor spec, a few degrees of margin taken off) - checked before send_angles.
JOINT_LIMITS_DEG = [(-165, 165), (-130, 130), (-145, 145), (-140, 140), (-160, 160), (-175, 175)]
HOME_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # arm straight up: safe start/end pose
SPEED = 30                                       # default servo speed (1-100)
SLOW_SPEED = 20                                   # calibration moves
# A move is finished when the arm stops moving; it then counts as reached if within these tolerances.
# (The arm typically stops 1-5 mm from a send_coords target, mostly in z: servo backlash / gravity sag.)
ANGLE_TOL_DEG = 3.0       # send_angles: every joint within this of the command
POS_TOL_MM = 10.0         # send_coords: get_coords within this of the target, else "unreachable"
MOVE_TIMEOUT_S = 15.0     # arm still moving after this long -> error

# Tool point: how far the "tip" sticks out below the flange (mm); the tool always points straight down,
# so tip = flange - (0, 0, TOOL_LENGTH_MM). Set it if you tape a pointer to the flange for calibration.
TOOL_LENGTH_MM = 20.0

# Starting overview pose for calibrate.py overview (flange coords, pointing straight down). The robot's own
# orientation reading at the saved overview pose is then reused for every move, so its Euler convention never
# has to be guessed. Pointing straight down, the arm only reaches roughly x = 160..240 mm: put the sheet centre
# near (x, y) = (200, -60) mm (y = -60 is straight ahead: the wrist has a 64.6 mm sideways offset).
DEFAULT_OVERVIEW_COORDS = [190.0, -60.0, 240.0, -180.0, 0.0, -90.0]

# --------------------------------------------------------------------------- task
HOVER_HEIGHT_MM = 40.0      # tool height above the table when hovering over a quadrant
APPROACH_EXTRA_MM = 50.0    # first go this much higher than the hover, then descend vertically
MIN_TOOL_HEIGHT_MM = 10.0   # never command the tool closer to the table than this (after calibration)

# --------------------------------------------------------------------------- vision
CAMERA_SIZE = (640, 480)
GESTURE_STABLE_FRAMES = 12   # same finger count this many frames in a row -> accepted
GESTURE_TIMEOUT_S = 30.0
SHEET_TIMEOUT_S = 15.0
SHEET_AVERAGE_FRAMES = 5     # average the sheet pose over this many frames with all 4 markers
