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
# file-name: ogm.py
# ========================================================================


import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import FaceLandmarker

from .actions import ActionType
from .camera_config import CameraConfig


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


from .face.face_engine import FaceLandmarksDetection
from .modules_base import Module

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")

class OGM:
    def __init__(
        self,
        modules: list[Module] | None = None,
        face_engine_object: FaceLandmarksDetection | None = None,
        # TODO: Add hand_engine_object when ready
    ) -> None:
        if modules is not None:
            self._modules: list[Module] = modules
        else:
            log.error("Modules are missing.")
            raise TypeError("Modules are missing.")

        self._face_engine_object: FaceLandmarksDetection | None = face_engine_object
        # Flag per indicare lo stato di esecuzione dell'API
        self._is_running: bool | None = None

        # Definizione del Lock Rientrante d'istanza
        self._rlock: threading.RLock = threading.RLock()

        # Definizione thread dell'engine mediapipe del volto
        self._face_engine_thread: threading.Thread

        self._actions: list[tuple[ActionType, int]] = []

        # Definizione del thread che chiama la callback on_action passandogli la copia
        # immutabile (tupla) della lista delle azioni
        self._actions_sentinel_thread: threading.Thread | None = None

        self._actions_queue: Queue[tuple[tuple[ActionType, int], ...] | None] = Queue()

        # Funzioni di callback
        self.on_action: Callable[[tuple[tuple[ActionType, int], ...]], None] | None = (
            None
        )
        self.on_calibration: Callable[[dict[str, dict[str, float]]], None] | None = None

    def _actions_sentinel(self) -> None:
        """
        Sentinel worker thread loop for invoking the `on_action` callback asynchronously.

        Continuously pops action sequence snapshots from `_actions_queue` and dispatches them to `on_blink`
        until a `None` sentinel item is received when stopping.
        """
        while self._is_running:
            actions: tuple[tuple[ActionType, int], ...] | None = (
                self._actions_queue.get()
            )
            if actions is None:
                break

            if self.on_action is not None:
                self.on_action(actions)

    def start(
        self,
        mode: str,
        camera_config: CameraConfig | None = None
    ) -> None:
        if self._is_running:
            log.error("The module has already been started.")
            return

        if mode == "calibrate":
            if self.on_calibration is None:
                log.error(
                    'The "on_calibration" callback function has not been defined before start.'
                )
                return

            self._is_running = True

            log.info("Avvio telecamera in modalità CALIBRAZIONE.")

        elif mode == "detect":
            # Implementare le actions
            if self.on_action is None:
                log.error(
                    'The "on_action" callback function has not been defined before start.'
                )
                return

            self._is_running = True
            self._actions_queue = Queue()

            self._actions_sentinel_thread = threading.Thread(
                target=self._actions_sentinel, daemon=True
            )
            self._actions_sentinel_thread.start()

        else:
            log.error(f'"{mode}" doesn\'t exist as an available mode.')
            return



        if self._face_engine_object is None:
            face_engine_object: FaceLandmarksDetection = FaceLandmarksDetection(modules=self._modules)
            self._face_engine_object = face_engine_object

        self._face_engine_object.on_calibration = self.on_calibration

        self._face_engine_thread = threading.Thread(
            target=self._face_engine_object.execution_loop,
            args=(mode, camera_config),
            daemon=True
        )

        self._face_engine_thread.start()
