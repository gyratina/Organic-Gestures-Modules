###
# project: Organic Gestures Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: __init__.py
###

from .blink_detector import ActionType, BlinkDetector
from .camera_config import CameraConfig

__all__ = ["BlinkDetector", "ActionType", "CameraConfig"]
