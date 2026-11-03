"""
Hand-guided teaching ("lead-through"): the servos are released, you move the arm by hand while watching the
live camera view, and a key press stiffens it exactly where you hold it.

Keys (click the camera window first):
    SPACE : release the servos  - HOLD THE ARM FIRST, it goes limp and will drop
    ENTER : lock the arm where it is and use this pose (also accepts the pose if the arm is already locked)
    ESC   : abort (the arm is locked first)

The arm is ALWAYS locked again before this function returns, even on errors or Ctrl+C.
"""

import time
from typing import Callable, List, Optional, Tuple

from common.camera import Camera, put_lines, show


def teach(robot, cam: Camera, title: str,
          overlay: Optional[Callable] = None) -> Optional[Tuple[List[float], List[float]]]:
    """
    Returns (coords, angles) read from the robot after locking when ENTER is pressed, or None on ESC.
    overlay(frame) may draw on the frame and returns extra text lines.
    """
    released = False
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            extra = overlay(frame) if overlay else []
            if released:
                state = ["ARM IS LIMP - move it by hand, keep holding it",
                         "ENTER = lock here and use this pose   ESC = abort"]
                color = (0, 0, 255)
            else:
                state = ["arm locked.  Hold the arm, then SPACE = release it for moving by hand",
                         "ENTER = use the current pose   ESC = abort"]
                color = (0, 255, 255)
            put_lines(frame, [title, *state, *extra], color=color)
            key = show(frame, "EE430 teach")

            if key == ord(" ") and not released:
                robot.release()
                released = True
            elif key == 13:
                if released:
                    robot.lock()
                    released = False
                    time.sleep(0.8)  # let the servos settle at the held pose
                return robot.get_coords(), robot.get_angles()
            elif key == 27:
                return None
    finally:
        if released:
            robot.lock()
