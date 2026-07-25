###
# project: Organic Gestures Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: camera_config.py
###

import logging

import cv2 as cv

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class CameraConfig:
    """
    Configuration manager for camera initialization and settings.
    """

    def __init__(
        self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30
    ) -> None:
        """
        Initializes the CameraConfig with desired video capture parameters.

        Args:
            camera_index (int): The index of the camera device (default is 0).
            width (int): The desired frame width (default is 1280).
            height (int): The desired frame height (default is 720).
            fps (int): The desired frames per second (default is 30).
        """
        self.camera_index: int = camera_index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps

    def set_camera(self) -> cv.VideoCapture:
        """
        Opens and configures the camera according to the initialization parameters.

        Returns:
            cv.VideoCapture: The configured OpenCV video capture object.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        video: cv.VideoCapture = cv.VideoCapture(self.camera_index)
        if not video.isOpened():
            log.error("Impossibile aprire la telecamera.\n")
            raise RuntimeError("Impossibile aprire la telecamera.")

        video.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=self.width)
        video.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=self.height)
        video.set(propId=cv.CAP_PROP_FPS, value=self.fps)

        return video
