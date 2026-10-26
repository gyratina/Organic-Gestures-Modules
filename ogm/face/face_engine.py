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
# file-name: face_engine.py
# ========================================================================

import logging
import os
import threading
import time
from collections.abc import Callable

import cv2 as cv
import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks import python as py
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)

from ogm.modules_base import Module

from ..camera_config import CameraConfig
from ..ogm import FrameData

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class FaceLandmarksDetection:
    def __init__(
        self,
        modules: list[Module],
        model_path: str | None = None
    ) -> None:
        # Flag per indicare lo stato di esecuzione dell'API
        self._is_running: bool | None = None

        # Face landmarker configuration
        self._face_landmarker: FaceLandmarker | None = None

        self._modules: list[Module] = modules

        # Percorso file del model bundle
        if model_path is None:
            package_directory: str = os.path.dirname(os.path.dirname(__file__))
            self._model_path: str = os.path.join(
                package_directory, "models", "face_landmarker.task"
            )
        else:
            self._model_path = model_path

        # Salva il timestamp dell'ultimo timestamp in millisecondi
        self._last_timestamp_ms: int = 0

        self._is_calibrating: bool = False
        self._calib_start_time: int | None = None
        self.on_calibration: Callable[[dict[str, dict[str, float]]], None] | None = None

        # TODO: Implementare la actions queue in base alla Classe ActionType da creare

        # Definizione del Lock Rientrante d'istanza
        self._rlock: threading.RLock = threading.RLock()


    def _frame_formattation(self, rgb: MatLike) -> None:
        """
        Formats an RGB frame for MediaPipe processing and submits it asynchronously.

        Wraps `rgb` into an `mp.Image` container, ensures frame timestamps (`frame_timestamp_ms`) are
        monotonically increasing to prevent MediaPipe timestamp errors, and invokes `detect_async`.

        Args:
            rgb (MatLike): The frame image formatted in RGB color space.
        """

        ### Fase di preparazione dati

        # Creazione oggetto mp.Image, che formatta i dati dei pixel in un formato compatibile con i modelli di MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calcolo del timestamp per ogni frame
        frame_timestamp_ms: int = time.monotonic_ns() // 1_000_000
        if frame_timestamp_ms <= self._last_timestamp_ms:
            frame_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = frame_timestamp_ms

        ### Fase di esecuzione
        if self._face_landmarker is not None:
            self._face_landmarker.detect_async(mp_image, frame_timestamp_ms)

    def _mediapipe_callback(
        self,
        result: FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        # Controllo se la telecamera ha trovato almeno un volto
        if not result.face_landmarks:
            return

        # Ottenimento dati sulla dimensione dell'immagine
        height, width = output_image.height, output_image.width

        # Vengono salvati i dati in merito alla prima faccia trovata da MediaPipe (478 oggetti NormalizedLandmark)
        face_landmarks = result.face_landmarks[0]

        frame_data: FrameData = FrameData(
            landmarks = face_landmarks,
            frame_height = height,
            frame_width = width,
            timestamp_ms = timestamp_ms,

            # TODO: Aggiungere i calcoli PnP per passare dati veri
            face_pitch = 0.0,
            face_yaw = 0.0,
            face_roll = 0.0,
            face_translation = 0.0
        )

        if self._is_calibrating and self._calib_start_time is None:
            self._calib_start_time = timestamp_ms
            log.info(
                "Inizio calibrazione: Guarda la telecamera con espressione neutra per 3 secondi.\n"
            )

        for module in self._modules:
            module._frame_processing(frame_data)
            if self._is_calibrating:
                module._calibration(frame_data)
            else:
                # TODO: Chiamare module._actions_timed_FSM(frame_data)
                pass

        if self._is_calibrating and self._calib_start_time is not None:
            time_elapsed: int = timestamp_ms - self._calib_start_time

            if time_elapsed >= 3000:
                calibration_data: dict[str, dict[str, float]] = {}
                for module in self._modules:
                    calibration_data[module.MODULE_NAME] = (
                        module._get_calibration_data()
                    )

                self._is_calibrating = False
                self._calib_start_time = None

                if self.on_calibration is not None:
                    self.on_calibration(calibration_data)


    def execution_loop(
        self,
        mode: str,
        # TODO: FARE CAM MULTI CANALE CON CONTROLLO IN OGM SE NONE
        camera_config: CameraConfig | None
    ) -> None:
        self._is_running = True
        self._is_calibrating = mode == "calibrate"

        # Impostazioni del modello di Landmarking facciale
        self._face_landmarker = FaceLandmarker.create_from_options(
            options=FaceLandmarkerOptions(
                base_options=py.BaseOptions(model_asset_path=self._model_path),
                running_mode=RunningMode.LIVE_STREAM,
                num_faces=1,
                result_callback=self._mediapipe_callback,
            )
        )

        if camera_config is None:
            camera_config = CameraConfig()

        video: cv.VideoCapture = camera_config.set_camera()

        while self._is_running:
            status, frame = video.read()

            if not status:
                log.error("Frame not found.")
                break

            rgb_frame: MatLike = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            self._frame_formattation(rgb=rgb_frame)

            if mode == "calibrate" and not self._is_calibrating:
                break

        self._is_running = False
        self._face_landmarker.close()
        video.release()
