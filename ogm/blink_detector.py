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
# file-name: blink_detector.py
# ========================================================================


import logging
import os
import threading
import time
from collections.abc import Callable
from enum import Enum
from math import dist
from typing import ClassVar, cast

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision

from .camera_config import CameraConfig

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class ActionType(Enum):
    """
    Enumeration representing the different types of detected ocular gestures.
    """

    LEFT = 0
    RIGHT = 1
    BOTH = 2


class BlinkDetector:
    """
    Core detector class that processes frames to identify and record voluntary ocular gestures (blinks).
    Supports single eye and both eyes gestures, featuring a state-machine based precision filter.
    """

    _LEFT_EYE: ClassVar[dict[str, int]] = {
        "P1": 263,  # Angolo esterno
        "P2": 385,  # Palpebra superiore
        "P3": 387,  # Palpebra superiore
        "P4": 362,  # Angolo interno
        "P5": 373,  # Palpebra inferiore
        "P6": 380,  # Palpebra inferiore
    }
    _RIGHT_EYE: ClassVar[dict[str, int]] = {
        "P1": 33,  # Angolo esterno
        "P2": 160,  # Palpebra superiore
        "P3": 158,  # Palpebra superiore
        "P4": 133,  # Angolo interno
        "P5": 153,  # Palpebra inferiore
        "P6": 144,  # Palpebra inferiore
    }
    _ORIENTATION_REFERENCES: ClassVar[dict[str, int]] = {
        "forehead": 9,
        "chin": 199,
        "left_corner": 263,
        "right_corner": 33,
    }

    # Metodo costruttore
    def __init__(
        self,
        base_left_ear_threshold: float = 0.16,
        base_right_ear_threshold: float = 0.16,
        min_blink_time_threshold: int = 80,
        max_blink_time_threshold: int = 500,
        max_combo_delay: int = 2000,
        ear_diff_ratio: float = 0.03,
        model_path: str | None = None,
        calibration_threshold_ratio: float = 0.50,
        sensitivity_coefficient: float = 0.4,
    ) -> None:
        """
        Initializes the BlinkDetector with specific thresholds for gesture detection.

        Args:
            left_ear_threshold (float): EAR threshold to consider the left eye closed.
            right_ear_threshold (float): EAR threshold to consider the right eye closed.
            min_blink_time_threshold (int): Minimum duration (ms) for a closure to be considered a voluntary blink.
            max_blink_time_threshold (int): Maximum duration (ms) for a closure to be considered a voluntary blink.
            max_combo_delay (int): Maximum delay (ms) allowed between consecutive blinks to be grouped into the same combo.
            ear_diff (float): Tolerance for EAR difference to avoid false asymmetrical blink triggers (e.g. skin pulling).
            model_path (str | None): Absolute path to the MediaPipe Face Landmarker model. If None, uses the bundled model.
            calibration_threshold_ratio (float): Ratio applied to the calculated EAR average during calibration (default is 0.60).
            sensitivity_coefficient (float): Coefficient to adjust the sensivity of the dynamic threshold based on face pitch.
        """
        # Flag per indicare lo stato di esecuzione dell'API
        self._is_running: bool | None = None

        # Salvataggio del thread per permettere al metodo close() di terminare l'esecuzione
        self._ogm_thread: threading.Thread | None = None

        # Soglie di apertura dell'occhio
        self._base_left_ear_threshold: float = base_left_ear_threshold
        self._base_right_ear_threshold: float = base_right_ear_threshold
        self._current_left_ear_threshold: float = self._base_left_ear_threshold
        self._current_right_ear_threshold: float = self._base_right_ear_threshold

        # Dizionari contente le coordinate X, Y, Z dei punti facciali utilizzato
        # in get_pixel_coordinates per convertire i punti facciali in coordinate
        # pixel
        self._pixelized_left_eye_dict: dict[str, tuple[float, float]] = {}
        self._pixelized_right_eye_dict: dict[str, tuple[float, float]] = {}
        self._pixelized_orientation_references_dict: dict[
            str, tuple[float, float, float]
        ] = {}

        # Parametri per la soglia dinamica in base all'inclinazione del volto
        self._delta_prospective_acc: float = 0.0
        self._current_delta_prospective: float | None = None
        self._default_face_prospective: float | None = None
        self._angular_delta: float = 0.0
        self._sensitivity_coefficient: float = sensitivity_coefficient

        # Soglia minima e massima per considerare il battito volontario
        self._min_blink_time_threshold: int = min_blink_time_threshold
        self._max_blink_time_threshold: int = max_blink_time_threshold

        self._max_combo_delay: int = max_combo_delay

        self._actions: list[tuple[ActionType, int]] = []
        self._last_reopening_timestamp: int | None = None

        self._min_floor_ratio: float = 0.88
        self._left_eye_min_floor: float = (
            self._base_left_ear_threshold * self._min_floor_ratio
        )
        self._right_eye_min_floor: float = (
            self._base_right_ear_threshold * self._min_floor_ratio
        )

        # Tolleranza della differenza di tipo EAR accettabile affinché si possa distinguere un occhio chiuso involontariamente
        # per il tiraggio della pelle nel tentativo di chiuderne uno solo
        self._ear_diff_ratio: float = ear_diff_ratio

        self._is_calibrating: bool = False
        self._calibration_threshold_ratio: float = calibration_threshold_ratio

        self._last_action: ActionType | None = None

        # Contatore del tempo per il quale l'occhio è stato chiuso
        self._blink_time_counter: int | None = None

        # Funzioni di callback
        self.on_blink: Callable[[list[tuple[ActionType, int]]], None] | None = None
        self.on_calibration: Callable[[float, float], None] | None = None
        self.telemetry_callback: Callable[[dict[str, float]], None] | None = None

        # Percorso file del model bundle
        if model_path is None:
            current_directory: str = os.path.dirname(__file__)
            self._model_path: str = os.path.join(
                current_directory, "models", "face_landmarker.task"
            )
        else:
            self._model_path = model_path

        # Salva il timestamp dell'ultimo timestamp in millisecondi
        self._last_timestamp_ms: int = 0

        # Variabili di conteggio e somma per la calibrazione degli occhi
        self._sum_left_ear: float = 0.0
        self._sum_right_ear: float = 0.0
        self._count_ear: int = 0
        self._calib_start_time: int | None = None

        # Face landmarker configuration
        self._face_landmarker: vision.FaceLandmarker | None = None

    def close(self) -> None:
        """
        Signals the internal execution loop to stop, which will smoothly release the camera and MediaPipe resources.
        This method will safely block (join) the calling thread until the background execution thread has completely terminated.
        """
        self._is_running = False
        if self._ogm_thread is not None:
            self._ogm_thread.join()
        log.info("Esecuzione modulo API terminata.")

    def reset_log(self) -> None:
        """
        Clears the logged actions and resets the combo timer. Usually called after a combo is successfully matched.
        """
        self._actions.clear()
        self._last_reopening_timestamp = None

    def _frame_preparation(self, rgb) -> None:
        """
        Prepares the frame for MediaPipe processing by converting formats and managing timestamps.

        Args:
            frame: The original BGR frame captured by OpenCV.
            rgb: The converted RGB frame to be used by MediaPipe.
        """

        ### Fase di preparazione dati

        # Creazione oggetto mp.Image, che formatta i dati dei pixel in un formato compatibile con i modelli di MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calcolo del timestamp per ogni frame
        frame_timestamp_ms: int = int(time.time() * 1000)
        if frame_timestamp_ms <= self._last_timestamp_ms:
            frame_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = frame_timestamp_ms

        ### Fase di esecuzione
        if self._face_landmarker is not None:
            self._face_landmarker.detect_async(mp_image, frame_timestamp_ms)

    def _get_pixel_coordinates(
        self,
        face_landmarks,
        frame_height: int,
        frame_width: int,
        points_dict: dict[str, int],
        pixel_points_dict: dict[str, tuple[float, ...]],
        is_face_orientation: bool = False,
    ) -> dict[str, tuple[float, ...]]:

        if not is_face_orientation:
            for key, index in points_dict.items():
                # Dai dati del volto, viene estratto il punto del volto equivalente all'indice iterato e assegnato a face_point_data.
                # Essendo che ogni punto del volto ha 3 attributi X, Y e Z adesso face_point_data ha i 3 punti di face_landmarks[index]
                face_point_data = face_landmarks[index]

                # A ogni iterazione viene aggiunto al nuovo dizionario il prodotto tra gli attributi delle coordinate X e Y di face_point_data
                # e la larghezza e altezza del frame.
                pixel_points_dict[key] = (
                    face_point_data.x * frame_width,
                    face_point_data.y * frame_height,
                )
        else:
            for key, index in points_dict.items():
                # Dai dati del volto, viene estratto il punto del volto equivalente all'indice iterato e assegnato a face_point_data.
                # Essendo che ogni punto del volto ha 3 attributi X, Y e Z adesso face_point_data ha i 3 punti di face_landmarks[index]
                face_point_data = face_landmarks[index]

                # Ricorda che la scala della profondità è proporzionale alla larghezza dell'immagine (fonte: docs di mediapipe)
                pixel_points_dict[key] = (
                    face_point_data.x * frame_width,
                    face_point_data.y * frame_height,
                    face_point_data.z * frame_width,
                )

        return pixel_points_dict

    def _precision_filter(
        self, sx_ear: float, dx_ear: float, timestamp_ms: int
    ) -> None:
        ### Calcolo soglia dinamica

        if (
            self._default_face_prospective is not None
            and self._current_delta_prospective is not None
        ):
            self._angular_delta = (
                self._current_delta_prospective - self._default_face_prospective
            )

            angular_delta_penality: float | None = None
            if self._angular_delta < 0:
                angular_delta_penality = (
                    0.7 * (self._angular_delta**2) * self._sensitivity_coefficient
                )
            else:
                angular_delta_penality = (
                    self._angular_delta**2 * self._sensitivity_coefficient
                )

            self._current_left_ear_threshold = self._base_left_ear_threshold - (
                angular_delta_penality
            )
            self._current_right_ear_threshold = self._base_right_ear_threshold - (
                angular_delta_penality
            )

            ### Clamping della soglia dinamica
            self._current_left_ear_threshold = max(
                self._left_eye_min_floor,
                min(self._base_left_ear_threshold, self._current_left_ear_threshold),
            )
            self._current_right_ear_threshold = max(
                self._right_eye_min_floor,
                min(self._base_right_ear_threshold, self._current_right_ear_threshold),
            )

        ### Filtro di precisione
        left_eye_ratio: float = sx_ear / self._current_left_ear_threshold
        right_eye_ratio: float = dx_ear / self._current_right_ear_threshold

        is_left_eye_closed: bool = left_eye_ratio < 1.0
        is_right_eye_closed: bool = right_eye_ratio < 1.0
        are_both_eyes_closed: bool = is_left_eye_closed and is_right_eye_closed

        reopening_moment: int | None = None
        lapse: int | None = None

        current_action: ActionType | None = None

        if are_both_eyes_closed:
            if (right_eye_ratio - left_eye_ratio) > self._ear_diff_ratio:
                current_action = ActionType.LEFT
            elif (left_eye_ratio - right_eye_ratio) > self._ear_diff_ratio:
                current_action = ActionType.RIGHT
            else:
                current_action = ActionType.BOTH

        elif current_action is None:
            if is_left_eye_closed:
                current_action = ActionType.LEFT
            elif is_right_eye_closed:
                current_action = ActionType.RIGHT
            else:
                current_action = None

        if self._blink_time_counter is None:
            self._blink_time_counter = timestamp_ms

        if current_action != self._last_action:
            if self._last_action is not None:
                reopening_moment = timestamp_ms
                blink_time: int = reopening_moment - self._blink_time_counter

                if (
                    self._min_blink_time_threshold
                    <= blink_time
                    <= self._max_blink_time_threshold
                ):
                    if not self._actions and self._last_reopening_timestamp is None:
                        self._actions.append((self._last_action, 0))
                        self._last_reopening_timestamp = reopening_moment

                    elif self._last_reopening_timestamp is not None:
                        lapse = (
                            self._blink_time_counter - self._last_reopening_timestamp
                        )
                        if lapse > self._max_combo_delay:
                            self._actions.clear()
                            self._actions.append((self._last_action, 0))
                            self._last_reopening_timestamp = reopening_moment
                        else:
                            self._actions[-1] = (self._actions[-1][0], lapse)
                            self._actions.append((self._last_action, 0))
                            self._last_reopening_timestamp = reopening_moment

                    # Chiamata a funzione di callback
                    if self.on_blink is not None:
                        self.on_blink(self._actions)

            self._last_action = current_action
            self._blink_time_counter = None

    def _mediapipe_callback(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """
        Callback invoked by MediaPipe for every processed frame.
        Handles coordinate extraction, dynamic threshold adjustments via pitch estimation,
        and delegates to the precision filter or calibration method.

        Args:
            result (vision.FaceLandmarkerResult): The face landmark detection results.
            output_image (mp.Image): The image processed by MediaPipe.
            timestamp_ms (int): The timestamp of the processed frame.
        """

        # Controllo se la telecamera ha trovato almeno un volto
        if not result.face_landmarks:
            return

        # Ottenimento dati sulla dimensione della camera
        height, width = output_image.height, output_image.width

        # Vengono salvati i dati in merito alla prima faccia trovata da MediaPipe (478 oggetti NormalizedLandmark)
        face_landmarks = result.face_landmarks[0]

        # Traduzione dei dizionari dei punti degli occhi in coordinate X, Y dei pixel sullo schermo
        left_eye_coordinates: dict[str, tuple[float, float]] = cast(
            dict[str, tuple[float, float]],
            self._get_pixel_coordinates(
                face_landmarks=face_landmarks,
                frame_height=height,
                frame_width=width,
                points_dict=self._LEFT_EYE,
                pixel_points_dict=self._pixelized_left_eye_dict,
            ),
        )
        right_eye_coordinates: dict[str, tuple[float, float]] = cast(
            dict[str, tuple[float, float]],
            self._get_pixel_coordinates(
                face_landmarks=face_landmarks,
                frame_height=height,
                frame_width=width,
                points_dict=self._RIGHT_EYE,
                pixel_points_dict=self._pixelized_right_eye_dict,
            ),
        )

        # Calcolo dell'EAR per l'occhio Sinistro (prospettiva umana)
        sx_ear: float = self._ear_math(
            eye_coordinates=left_eye_coordinates,
        )

        # Calcolo dell'EAR per l'occhio Destro (prospettiva umana)
        dx_ear: float = self._ear_math(
            eye_coordinates=right_eye_coordinates,
        )

        orientation_coordinates: dict[str, tuple[float, float, float]] = cast(
            dict[str, tuple[float, float, float]],
            self._get_pixel_coordinates(
                face_landmarks=face_landmarks,
                frame_height=height,
                frame_width=width,
                points_dict=self._ORIENTATION_REFERENCES,
                pixel_points_dict=self._pixelized_orientation_references_dict,
                is_face_orientation=True,
            ),
        )

        self._current_delta_prospective = (
            orientation_coordinates["chin"][2] - orientation_coordinates["forehead"][2]
        ) / dist(
            orientation_coordinates["left_corner"],
            orientation_coordinates["right_corner"],
        )

        # Se la calibrazione è impostata su True allora avvia la calibrazione,
        # altrimenti continua filtrando le gesture e chiamando funzioni di callback
        if self._is_calibrating:
            self.calibration(sx_ear=sx_ear, dx_ear=dx_ear, timestamp_ms=timestamp_ms)
        else:
            self._precision_filter(
                sx_ear=sx_ear, dx_ear=dx_ear, timestamp_ms=timestamp_ms
            )

        telemetry_dict: dict[str, float] = {
            "sx_eye_EAR": sx_ear,
            "dx_eye_EAR": dx_ear,
            "sx_eye_EAR_THRESHOLD": self._current_left_ear_threshold,
            "dx_eye_EAR_THRESHOLD": self._current_right_ear_threshold,
            "face_pitch": self._angular_delta,
        }

        if self.telemetry_callback is not None:
            self.telemetry_callback(telemetry_dict)

    def _ear_math(
        self,
        eye_coordinates: dict[str, tuple[float, float]],
    ) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR) based on the 6 facial landmarks defining an eye.

        Args:
            eye_coordinates (dict): A dictionary mapping 'P1' through 'P6' to numpy coordinate arrays.

        Returns:
            float: The computed EAR value for the given eye.
        """
        # Aliases
        P1 = eye_coordinates["P1"]
        P2 = eye_coordinates["P2"]
        P3 = eye_coordinates["P3"]
        P4 = eye_coordinates["P4"]
        P5 = eye_coordinates["P5"]
        P6 = eye_coordinates["P6"]

        # Calcolo EAR (Eye Aspect Ratio)
        numerator: float = dist(P2, P6) + dist(P3, P5)
        denominator: float = 2 * dist(P1, P4)
        EAR: float = numerator / denominator

        return EAR

    def calibration(self, sx_ear: float, dx_ear: float, timestamp_ms: int) -> None:
        """
        Calibrates the base EAR thresholds and default face pitch (prospective) over a 3-second period.

        Args:
            sx_ear (float): The current EAR for the left eye.
            dx_ear (float): The current EAR for the right eye.
            timestamp_ms (int): The current frame timestamp in milliseconds.
        """
        if self._calib_start_time is None:
            self._calib_start_time = timestamp_ms
            log.info(
                "Inizio calibrazione: Guarda la telecamera con espressione neutra per 3 secondi.\n"
            )

        self._sum_left_ear += sx_ear
        self._sum_right_ear += dx_ear

        if self._current_delta_prospective is not None:
            self._delta_prospective_acc += self._current_delta_prospective

        self._count_ear += 1

        time_elapsed: int = timestamp_ms - self._calib_start_time

        if time_elapsed >= 3000:
            AVG_LEFT_EAR: float = self._sum_left_ear / self._count_ear
            AVG_RIGHT_EAR: float = self._sum_right_ear / self._count_ear

            LEFT_EAR_THRESHOLD: float = AVG_LEFT_EAR * self._calibration_threshold_ratio
            RIGHT_EAR_THRESHOLD: float = (
                AVG_RIGHT_EAR * self._calibration_threshold_ratio
            )

            self._is_calibrating = False
            self._calib_start_time = None

            self._default_face_prospective = (
                self._delta_prospective_acc / self._count_ear
            )

            self._base_left_ear_threshold = LEFT_EAR_THRESHOLD
            self._base_right_ear_threshold = RIGHT_EAR_THRESHOLD

            if self.on_calibration is not None:
                self.on_calibration(LEFT_EAR_THRESHOLD, RIGHT_EAR_THRESHOLD)

                self._left_eye_min_floor = (
                    self._base_left_ear_threshold * self._min_floor_ratio
                )
                self._right_eye_min_floor = (
                    self._base_right_ear_threshold * self._min_floor_ratio
                )

    def _execution_loop(
        self, mode: str = "detect", camera_config: CameraConfig | None = None
    ) -> None:
        """
        Internal method that runs the camera loop and processes frames synchronously.
        This is automatically executed in a background daemon thread by start().
        It initializes the MediaPipe FaceLandmarker natively within this thread to ensure memory safety
        and avoid threading issues upon restart, and resets calibration accumulators to prevent stale data.

        Args:
            mode (str): Operational mode. Use "calibrate" for threshold calibration or "detect" for gesture recognition.
            camera_config (CameraConfig | None): Custom camera configuration. If None, default 720p 30fps config is used.
        """
        self._is_running = True

        # Impostazioni del modello di Landmarking facciale
        BaseOptions = py.BaseOptions
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self._mediapipe_callback,
        )
        FaceLandmarker = vision.FaceLandmarker

        self._face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions
        )

        match mode:
            case "calibrate":
                self._is_calibrating = True
                self._calib_start_time = None

                self._sum_left_ear = 0.0
                self._sum_right_ear = 0.0
                self._count_ear = 0
                self._delta_prospective_acc = 0.0

                log.info("Avvio telecamera in modalità CALIBRAZIONE.")
            case _:
                self._is_calibrating = False
                log.info("Avvio telecamera in modalità RILEVAMENTO.")

        if camera_config is None:
            camera_config = CameraConfig()

        video: cv.VideoCapture = camera_config.set_camera()

        while self._is_running:
            status, frame = video.read()

            if not status:
                log.error("Errore, impossibile trovare un fotogramma.")
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            self._frame_preparation(rgb=rgb_frame)

            if mode == "calibrate" and self._is_calibrating is False:
                break

        self._face_landmarker.close()
        video.release()

    def start(
        self, mode: str = "detect", camera_config: CameraConfig | None = None
    ) -> None:
        """
        Starts the gesture detection asynchronously in a background daemon thread.
        Does not block the main thread. The thread instance is saved in `self.ogm_thread`
        so it can be properly joined by the `close()` method to prevent segmentation faults.

        Args:
            mode (str): Operational mode. Use "calibrate" for threshold calibration or "detect" for gesture recognition.
            camera_config (CameraConfig | None): Custom camera configuration. If None, default 720p 30fps config is used.
        """
        self._ogm_thread = threading.Thread(
            target=self._execution_loop, args=(mode, camera_config), daemon=True
        )
        self._ogm_thread.start()
