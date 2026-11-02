"""
Finds which laptop camera index is the flange camera: opens indices 0-5 and shows each for a few seconds.
    python project/list_cameras.py
Press any key in the window to go to the next camera, q to quit.
"""
import cv2

for idx in range(6):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        continue
    print(f"Camera index {idx}: opened ({int(cap.get(3))}x{int(cap.get(4))}) - press any key for next, q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"index {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("camera probe", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            raise SystemExit
        if key != 255:
            break
    cap.release()
cv2.destroyAllWindows()
