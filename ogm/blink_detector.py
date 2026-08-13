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
from queue import Queue
from typing import ClassVar, cast

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

from .camera_config import CameraConfig

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class ActionType(Enum):
    """
    Enumeration representing the different types of detected ocular gestures.

    Members:
        LEFT (int): Voluntary blink of the left eye (human perspective).
        RIGHT (int): Voluntary blink of the right eye (human perspective).
        BOTH (int): Simultaneous voluntary blink of both eyes.
    """

    LEFT = 0
    RIGHT = 1
    BOTH = 2


class BlinkDetector:
    """
    Core detector class that processes camera frames to identify and record voluntary ocular gestures.

    Features a state-machine precision filter, monocular vs binocular blink differentiation
    handling skin-pull effects (`ear_diff_ratio`), dynamic threshold scaling based on head pitch
    estimation, automatic 3-second calibration mode, and thread-safe asynchronous sentinel callbacks.

    Attributes:
        on_blink (Callable[[tuple[tuple[ActionType, int], ...]], None] | None): Callback executed when
            a valid blink sequence is recognized. Receives an immutable tuple of `(ActionType, pause_ms)`.
        on_calibration (Callable[[float, float], None] | None): Callback executed upon completing
            automatic calibration. Receives `(left_ear_threshold, right_ear_threshold)`.
        telemetry_callback (Callable[[dict[str, float]], None] | None): Callback executed on every frame
            when telemetry is enabled. Receives frame metrics (EAR values, thresholds, head pitch).
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
        ear_diff_ratio: float = 0.20,
        model_path: str | None = None,
        calibration_threshold_ratio: float = 0.50,
        sensitivity_coefficient: float = 0.05,
    ) -> None:
        """
        Initializes the BlinkDetector with specific parameters for gesture recognition and threshold adaptation.

        Args:
            base_left_ear_threshold (float): Baseline EAR threshold below which the left eye is considered closed (default: 0.16).
            base_right_ear_threshold (float): Baseline EAR threshold below which the right eye is considered closed (default: 0.16).
            min_blink_time_threshold (int): Minimum eye closure duration in ms to qualify as a voluntary blink (default: 80).
            max_blink_time_threshold (int): Maximum eye closure duration in ms to qualify as a voluntary blink (default: 500).
            max_combo_delay (int): Maximum inter-blink delay in ms permitted before resetting the logged combo sequence (default: 2000).
            ear_diff_ratio (float): Relative EAR closure difference threshold to distinguish monocular winks from binocular blinks (default: 0.20).
            model_path (str | None): Absolute path to the MediaPipe `face_landmarker.task` model file. If None, uses the bundled model asset.
            calibration_threshold_ratio (float): Ratio applied to average open-eye EAR observed during 3s calibration (default: 0.50).
            sensitivity_coefficient (float): Coefficient scaling the dynamic EAR threshold penalty based on face pitch angle deviation (default: 0.05).
        """
        # Flag per indicare lo stato di esecuzione dell'API
        self._is_running: bool | None = None

        # Coda Thread-Safe per passare le azioni al thread che chiama la callback on_action
        self._actions_queue: Queue[tuple[tuple[ActionType, int], ...] | None] = Queue()
        self._telemetry_queue: Queue[dict[str, float] | None] = Queue()

        # Definizione del Lock Rientrante d'istanza
        self._rlock: threading.RLock = threading.RLock()

        # Definizione del thread del modulo
        self._ogm_thread: threading.Thread | None = None

        # Definizione del thread che chiama la callback on_blink passandogli la copia
        # immutabile (tupla) della lista delle azioni
        self._actions_sentinel_thread: threading.Thread | None = None

        # Definizione del thread che chiama la callback telemetry_callback in modo
        # thread-safe
        self._telemetry_sentinel_thread: threading.Thread | None = None

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
        self.on_blink: Callable[[tuple[tuple[ActionType, int], ...]], None] | None = (
            None
        )
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
        self._face_landmarker: FaceLandmarker | None = None

    def close(self) -> None:
        """
        Signals background threads to stop and cleanly releases OpenCV and MediaPipe resources.

        Pushes termination sentinels (`None`) to internal queues, then safely blocks (joins) the calling
        thread for up to 2.0 seconds per background thread (`_ogm_thread`, `_actions_sentinel_thread`,
        and `_telemetry_sentinel_thread`) to prevent thread leakage and segmentation faults.
        """
        self._is_running = False
        self._actions_queue.put(item=None)
        self._telemetry_queue.put(item=None)

        # Controllo dell'esistenza del thread e se è ancora in esecuzione
        # Se non fosse in esecuzione sarebbe inutile fare il join
        if self._ogm_thread is not None and self._ogm_thread.is_alive():
            self._ogm_thread.join(timeout=2.0)

        if (
            self._actions_sentinel_thread is not None
            and self._actions_sentinel_thread.is_alive()
        ):
            self._actions_sentinel_thread.join(timeout=2.0)
            self._actions_queue = Queue()

        if (
            self._telemetry_sentinel_thread is not None
            and self._telemetry_sentinel_thread.is_alive()
        ):
            self._telemetry_sentinel_thread.join(timeout=2.0)
            self._telemetry_queue = Queue()

        log.info("Esecuzione modulo di blinking terminata.")

    def reset_log(self) -> None:
        """
        Clears the logged action sequence buffer and resets inter-blink combo timers.

        Acquires the internal reentrant lock (`_rlock`) to ensure thread-safe clearing of `_actions`
        and resetting of `_last_reopening_timestamp`. Typically invoked after matching a desired combo.
        """
        with self._rlock:
            self._actions.clear()
            self._last_reopening_timestamp = None

    def _frame_preparation(self, rgb: MatLike) -> None:
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
        frame_timestamp_ms: int = int(time.time() * 1000)
        if frame_timestamp_ms <= self._last_timestamp_ms:
            frame_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = frame_timestamp_ms

        ### Fase di esecuzione
        if self._face_landmarker is not None:
            self._face_landmarker.detect_async(mp_image, frame_timestamp_ms)

    def _get_pixel_coordinates(
        self,
        face_landmarks,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        frame_height: int,
        frame_width: int,
        points_dict: dict[str, int],
        pixel_points_dict: dict[str, tuple[float, ...]],
        is_face_orientation: bool = False,
    ) -> dict[str, tuple[float, ...]]:
        """
        Converts normalized MediaPipe landmark coordinates [0.0, 1.0] into screen pixel coordinates.

        Args:
            face_landmarks: Normalized landmarks list returned by MediaPipe FaceLandmarker for a detected face.
            frame_height (int): Height of the captured video frame in pixels.
            frame_width (int): Width of the captured video frame in pixels.
            points_dict (dict[str, int]): Dictionary mapping landmark key names to MediaPipe mesh indices.
            pixel_points_dict (dict[str, tuple[float, ...]]): Target dictionary updated in-place with pixel coordinates.
            is_face_orientation (bool): If True, computes 3D coordinates `(x*W, y*H, z*W)` for pitch calculations.
                If False, computes 2D coordinates `(x*W, y*H)`.

        Returns:
            dict[str, tuple[float, ...]]: Updated dictionary mapping landmark keys to 2D or 3D pixel coordinate tuples.
        """

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
        self, SX_EAR: float, DX_EAR: float, timestamp_ms: int
    ) -> None:
        """
        Evaluates frame EAR values against dynamic thresholds, state-machine closure timing, and combo logic.

        Computes face pitch angle penalties to dynamically lower EAR thresholds (`_current_left_ear_threshold`,
        `_current_right_ear_threshold`), enforces minimum floor clamping, classifies current eye state
        (LEFT, RIGHT, BOTH) taking `ear_diff_ratio` into account for skin-pull tolerance, filters voluntary
        blink durations (`min_blink_time_threshold` to `max_blink_time_threshold`), updates action history
        with inter-blink pause intervals (`lapse` vs `max_combo_delay`), and pushes action sequence snapshots
        to `_actions_queue`.

        Args:
            sx_ear (float): Calculated Eye Aspect Ratio for the left eye.
            dx_ear (float): Calculated Eye Aspect Ratio for the right eye.
            timestamp_ms (int): Current frame timestamp in milliseconds.
        """
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
        left_eye_ratio: float = SX_EAR / self._current_left_ear_threshold
        right_eye_ratio: float = DX_EAR / self._current_right_ear_threshold

        is_left_eye_closed: bool = left_eye_ratio <= 1.0
        is_right_eye_closed: bool = right_eye_ratio <= 1.0
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
                    with self._rlock:
                        if not self._actions and self._last_reopening_timestamp is None:
                            self._actions.append((self._last_action, 0))
                            self._last_reopening_timestamp = reopening_moment

                        elif self._last_reopening_timestamp is not None:
                            lapse = (
                                self._blink_time_counter
                                - self._last_reopening_timestamp
                            )
                            if lapse > self._max_combo_delay:
                                self._actions.clear()
                                self._actions.append((self._last_action, 0))
                                self._last_reopening_timestamp = reopening_moment
                            else:
                                self._actions[-1] = (self._actions[-1][0], lapse)
                                self._actions.append((self._last_action, 0))
                                self._last_reopening_timestamp = reopening_moment

                        # Creazione copia (snapshot) immutabile (tuple) della lista _actions
                        snapshot: tuple[tuple[ActionType, int], ...] = tuple(
                            self._actions
                        )

                    # Passaggio dello snapshot di _actions alla coda thread-safe che chiama la
                    # callback su un thread separato
                    self._actions_queue.put(item=snapshot)

            self._last_action = current_action
            self._blink_time_counter = None

    def _mediapipe_callback(
        self,
        result: FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """
        MediaPipe `detect_async` callback invoked for every analyzed frame.

        Extracts face landmarks, calculates left (`sx_ear`) and right (`dx_ear`) eye aspect ratios,
        computes relative face pitch ratio `(z_chin - z_forehead) / dist(left_corner, right_corner)`,
        routes execution to `calibration` or `_precision_filter`, and puts telemetry metadata into `_telemetry_queue`.

        Args:
            result (FaceLandmarkerResult): Landmarks detection result returned by MediaPipe.
            output_image (mp.Image): Image container processed by MediaPipe.
            timestamp_ms (int): Monotonic frame timestamp in milliseconds.
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

        # Calcolo dell'EAR per l'occhio Sinistro (prospettiva humana)
        SX_EAR: float = self._ear_math(
            eye_coordinates=left_eye_coordinates,
        )

        # Calcolo dell'EAR per l'occhio Destro (prospettiva humana)
        DX_EAR: float = self._ear_math(
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
            self.calibration(sx_ear=SX_EAR, dx_ear=DX_EAR, timestamp_ms=timestamp_ms)
        else:
            self._precision_filter(
                SX_EAR=SX_EAR, DX_EAR=DX_EAR, timestamp_ms=timestamp_ms
            )

        telemetry_dict: dict[str, float] = {
            "sx_eye_EAR": SX_EAR,
            "dx_eye_EAR": DX_EAR,
            "sx_eye_EAR_THRESHOLD": self._current_left_ear_threshold,
            "dx_eye_EAR_THRESHOLD": self._current_right_ear_threshold,
            "face_pitch": self._angular_delta,
        }

        if self.telemetry_callback is not None:
            self._telemetry_queue.put(item=telemetry_dict)

    def _ear_math(
        self,
        eye_coordinates: dict[str, tuple[float, float]],
    ) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR) from 6 landmark screen pixel coordinates.

        Implements the Soukupová Eye Aspect Ratio formula:
        `EAR = (dist(P2, P6) + dist(P3, P5)) / (2 * dist(P1, P4))`

        Args:
            eye_coordinates (dict[str, tuple[float, float]]): Mapping of landmark keys ('P1' through 'P6')
                to 2D screen pixel coordinate tuples `(x, y)`.

        Returns:
            float: The computed non-dimensional EAR value for the specified eye.
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
        Executes baseline EAR threshold and default face pitch calibration over a 3-second (3000 ms) window.

        Accumulates open-eye EAR metrics and pitch ratios, calculates open-eye averages after 3000 ms,
        computes baseline closure thresholds (`AVG_EAR * calibration_threshold_ratio`), initializes minimum threshold
        floors, updates `_default_face_prospective`, disables calibration mode, and triggers `on_calibration`.

        Args:
            sx_ear (float): Instantaneous EAR for the left eye.
            dx_ear (float): Instantaneous EAR for the right eye.
            timestamp_ms (int): Current frame timestamp in milliseconds.
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

            self._left_eye_min_floor = (
                self._base_left_ear_threshold * self._min_floor_ratio
            )
            self._right_eye_min_floor = (
                self._base_right_ear_threshold * self._min_floor_ratio
            )

            if self.on_calibration is not None:
                self.on_calibration(LEFT_EAR_THRESHOLD, RIGHT_EAR_THRESHOLD)

    def _actions_sentinel(self) -> None:
        """
        Sentinel worker thread loop for invoking the `on_blink` callback asynchronously.

        Continuously pops action sequence snapshots from `_actions_queue` and dispatches them to `on_blink`
        until a `None` sentinel item is received when stopping.
        """
        while self._is_running:
            actions: tuple[tuple[ActionType, int], ...] | None = (
                self._actions_queue.get()
            )
            if actions is None:
                break

            if self.on_blink is not None:
                self.on_blink(actions)

    def _telemetry_sentinel(self) -> None:
        """
        Sentinel worker thread loop for invoking `telemetry_callback` asynchronously.

        Continuously pops telemetry metric dictionaries from `_telemetry_queue` and dispatches them to
        `telemetry_callback` until a `None` sentinel item is received when stopping.
        """
        while self._is_running:
            telemetry_dict: dict[str, float] | None = self._telemetry_queue.get()

            if telemetry_dict is None:
                break

            if self.telemetry_callback is not None:
                self.telemetry_callback(telemetry_dict)

    def _execution_loop(
        self, mode: str = "detect", camera_config: CameraConfig | None = None
    ) -> None:
        """
        Main execution loop running video capture and MediaPipe landmark detection in a daemon thread.

        Instantiates `FaceLandmarker` natively within the thread context to ensure thread memory safety,
        opens the camera via `camera_config.set_camera()`, loops through captured frames while `_is_running`
        is True, and releases OpenCV and MediaPipe resources upon loop exit or calibration completion.

        Args:
            mode (str): Operational mode. Use "calibrate" for threshold calibration or "detect" for gesture recognition.
            camera_config (CameraConfig | None): Custom camera configuration. If None, default 720p 30fps config is used.
        """

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
                log.error("Errore, impossibile trovare un fotogramma.")
                break

            rgb_frame: MatLike = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            self._frame_preparation(rgb=rgb_frame)

            if mode == "calibrate" and self._is_calibrating is False:
                break

        self._face_landmarker.close()
        video.release()

    def start(
        self, mode: str = "detect", camera_config: CameraConfig | None = None
    ) -> None:
        """
        Starts gesture detection or calibration asynchronously in a background daemon thread.

        Validates required callbacks (`on_calibration` for "calibrate" mode, `on_blink` for "detect" mode),
        initializes communication queues, launches sentinel threads (`_actions_sentinel_thread`,
        `_telemetry_sentinel_thread`), and spawns `self._ogm_thread` running `_execution_loop`.

        Args:
            mode (str): Operational mode. Use "calibrate" for threshold calibration or "detect" for gesture recognition.
            camera_config (CameraConfig | None): Custom camera configuration. If None, default 720p 30fps config is used.
        """
        if self._is_running:
            log.error("The module has already been started.")
            return

        match mode:
            case "calibrate":
                if self.on_calibration is None:
                    log.error(
                        'The "on_calibration" callback function has not been defined before start.'
                    )
                    return

                self._is_running = True
                self._is_calibrating = True
                self._calib_start_time = None

                self._sum_left_ear = 0.0
                self._sum_right_ear = 0.0
                self._count_ear = 0
                self._delta_prospective_acc = 0.0

                log.info("Avvio telecamera in modalità CALIBRAZIONE.")

            case _:
                if self.on_blink is None:
                    log.error(
                        'The "on_blink" callback function has not been defined before start.'
                    )
                    return

                self._is_running = True
                self._is_calibrating = False
                self._actions_queue = Queue()

                self._actions_sentinel_thread = threading.Thread(
                    target=self._actions_sentinel, daemon=True
                )
                self._actions_sentinel_thread.start()

                log.info("Avvio telecamera in modalità RILEVAMENTO.")

        if self.telemetry_callback is not None:
            self._telemetry_queue = Queue()

            self._telemetry_sentinel_thread = threading.Thread(
                target=self._telemetry_sentinel, daemon=True
            )
            self._telemetry_sentinel_thread.start()
        else:
            log.info("Telemetry functionalities are disabled.")

        self._ogm_thread = threading.Thread(
            target=self._execution_loop, args=(mode, camera_config), daemon=True
        )

        # Viene fatto partire prima il thread delle azioni per evitare di perdere eventuali prime azioni
        self._ogm_thread.start()
