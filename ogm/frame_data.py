from dataclasses import dataclass

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


@dataclass(slots=True)
class FrameData:
    landmarks: list[NormalizedLandmark]
    frame_height: int
    frame_width: int
    timestamp_ms: int
    face_pitch: float
    face_yaw: float
    face_roll: float
    face_translation: float
