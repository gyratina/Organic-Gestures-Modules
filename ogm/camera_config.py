# ========================================================================
# Copyright 2026 Valerio Di Tommaso (@gyratina)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
# project: Organic Gestures Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina (on GitHub)
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: camera_config.py
# ========================================================================


import logging

import cv2 as cv

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class CameraConfig:
    """
    Configuration manager for camera device initialization and hardware settings.

    Stores capture properties such as device index, target frame dimensions, and FPS,
    and handles opening and setting properties on the OpenCV `VideoCapture` object.

    Attributes:
        camera_index (int): Index of the video capture device (e.g., 0 for default webcam).
        width (int): Target frame width in pixels.
        height (int): Target frame height in pixels.
        fps (int): Target frame capture rate in frames per second.
    """

    def __init__(
        self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30
    ) -> None:
        """
        Initializes the CameraConfig with desired video capture parameters.

        Args:
            camera_index (int): The index of the target camera device (default is 0).
            width (int): The target frame width in pixels (default is 1280).
            height (int): The target frame height in pixels (default is 720).
            fps (int): The target capture rate in frames per second (default is 30).
        """
        self.camera_index: int = camera_index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps

    def set_camera(self) -> cv.VideoCapture:
        """
        Opens the camera device and configures width, height, and FPS properties.

        Applies `cv.CAP_PROP_FRAME_WIDTH`, `cv.CAP_PROP_FRAME_HEIGHT`, and
        `cv.CAP_PROP_FPS` settings to the initialized OpenCV `VideoCapture` object.

        Returns:
            cv.VideoCapture: The configured OpenCV video capture instance.

        Raises:
            RuntimeError: If the specified camera device index cannot be opened.
        """
        video: cv.VideoCapture = cv.VideoCapture(self.camera_index)
        if not video.isOpened():
            log.error("Impossibile aprire la telecamera.\n")
            raise RuntimeError("Impossibile aprire la telecamera.")

        video.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=self.width)
        video.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=self.height)
        video.set(propId=cv.CAP_PROP_FPS, value=self.fps)

        return video
