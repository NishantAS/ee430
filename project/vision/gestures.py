"""Factory for the two interchangeable finger counters (both expose .count(frame), .draw(frame), .latency_ms)."""

CLASSIFIERS = ("mediapipe", "classical")


def make_counter(name: str):
    if name == "classical":
        from vision.gesture_classical import ClassicalGestureCounter
        return ClassicalGestureCounter()
    if name == "mediapipe":
        from dl.gesture_mediapipe import MediaPipeGestureCounter  # imported lazily: heavy dependency
        return MediaPipeGestureCounter()
    raise ValueError(f"unknown classifier {name!r}, choose from {CLASSIFIERS}")
