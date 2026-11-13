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
# file-name: blinks_module.py
# ========================================================================


import logging
from math import dist
from typing import ClassVar, override

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from ogm.actions import ActionsContext, BlinkAction

from ..frame_data import FrameData
from ..modules_base import Module

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class BlinksModule(Module):
    MODULE_NAME: ClassVar[str] = "blink_mod"

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

    def __init__(
        self,
        base_left_ear_threshold: float = 0.16,
        base_right_ear_threshold: float = 0.16,
        min_blink_time_threshold: int = 50,
        max_blink_time_threshold: int = 1000,
        max_combo_delay: int = 2000, # TODO: Spostarlo in ActionsContext
        ear_diff_ratio: float = 0.20,
        calibration_threshold_ratio: float = 0.50,
    ) -> None:
        # Dizionari con salvati coordinate 2D dei pixel della camera
        # corrispondenti ai 6 Mediapipe Face Landmarks per occhio
        self.left_eye_coordinates: dict[str, tuple[float, float]] = {}
        self.right_eye_coordinates: dict[str, tuple[float, float]] = {}

        # Valori EAR NON ottimizzati (pre-EMA) del frame corrente
        self._raw_sx_ear: float = 0.0
        self._raw_dx_ear: float = 0.0

        # Valori EAR ottimizzati (post-EMA) del frame corrente
        self._post_ema_sx_ear: float = 0.0
        self._post_ema_dx_ear: float = 0.0

        # Valori EAR ottimizzati (post-EMA) del frame precedente
        self._previous_left_ear: float | None = None
        self._previous_right_ear: float | None = None

        # Soglie di apertura degli occhi di default
        self._base_left_ear_threshold: float = base_left_ear_threshold
        self._base_right_ear_threshold: float = base_right_ear_threshold
        self._calibration_threshold_ratio: float = calibration_threshold_ratio

        # Soglie di apertura degli occhi dinamica (si ribilancia a ogni frame se necessario)
        self._current_left_ear_threshold: float = self._base_left_ear_threshold
        self._current_right_ear_threshold: float = self._base_right_ear_threshold

        # Soglia minima e massima per considerare il battito di palpebre volontario
        self._min_blink_time_threshold: int = min_blink_time_threshold
        self._max_blink_time_threshold: int = max_blink_time_threshold

        # Variabili di conteggio e somma per la calibrazione degli occhi
        self._sum_left_ear: float = 0.0
        self._sum_right_ear: float = 0.0
        self._count_ear: int = 0

        # Tolleranza della differenza di tipo EAR accettabile affinché si possa distinguere un occhio chiuso involontariamente
        # per il tiraggio della pelle nel tentativo di chiuderne uno solo
        self._ear_diff_ratio: float = ear_diff_ratio

        # Salvataggio dell'ultimo stato della FSM nel FRAME precedente
        self._last_state: BlinkAction | None = None

        # Timestamp in millisecondi dell'inizio dello stato del FRAME precedente
        self._last_state_timestamp_ms: int | None = None



    @override
    def _frame_processing(self, frame_data: FrameData) -> None:
        self._get_coordinates(
            face_landmarks=frame_data.landmarks,
            frame_height=frame_data.frame_height,
            frame_width=frame_data.frame_width,
            highlight_points=self._LEFT_EYE,
            output_dict=self.left_eye_coordinates,
        )
        self._get_coordinates(
            face_landmarks=frame_data.landmarks,
            frame_height=frame_data.frame_height,
            frame_width=frame_data.frame_width,
            highlight_points=self._RIGHT_EYE,
            output_dict=self.right_eye_coordinates,
        )

        self._raw_sx_ear = self._calcs(
            highlight_coordinates=self.left_eye_coordinates,
        )
        self._raw_dx_ear = self._calcs(
            highlight_coordinates=self.right_eye_coordinates,
        )

    @override
    def _calibration(self, frame_data: FrameData) -> None:
        self._sum_left_ear += self._raw_sx_ear
        self._sum_right_ear += self._raw_dx_ear
        self._count_ear += 1

    @override
    def _get_calibration_data(self) -> dict[str, float]:
        if self._count_ear > 0:
            avg_left_ear: float = self._sum_left_ear / self._count_ear
            avg_right_ear: float = self._sum_right_ear / self._count_ear

            self._base_left_ear_threshold = (
                avg_left_ear * self._calibration_threshold_ratio
            )
            self._base_right_ear_threshold = (
                avg_right_ear * self._calibration_threshold_ratio
            )

        self._sum_left_ear = 0.0
        self._sum_right_ear = 0.0
        self._count_ear = 0

        return {
            "left_ear_threshold": self._base_left_ear_threshold,
            "right_ear_threshold": self._base_right_ear_threshold,
        }

    @override
    def _get_coordinates(
        self,
        face_landmarks: list[NormalizedLandmark],
        frame_height: int,
        frame_width: int,
        highlight_points: dict[str, int],
        output_dict: dict[str, tuple[float, float]],
    ) -> None:
        for key, index in highlight_points.items():
            # Dai dati del volto, viene estratto il punto del volto equivalente all'indice iterato e assegnato a face_point_data.
            # Essendo che ogni punto del volto ha 3 attributi X, Y e Z adesso face_point_data ha i 3 punti di face_landmarks[index]
            face_point_data = face_landmarks[index]

            # A ogni iterazione viene aggiunto al nuovo dizionario il prodotto tra gli attributi delle coordinate X e Y di face_point_data
            # e la larghezza e altezza del frame.
            if face_point_data.x is not None and face_point_data.y is not None:
                output_dict[key] = (
                    face_point_data.x * frame_width,
                    face_point_data.y * frame_height,
                )

    @override
    def _calcs(self, highlight_coordinates: dict[str, tuple[float, float]]) -> float:
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
        P1 = highlight_coordinates["P1"]
        P2 = highlight_coordinates["P2"]
        P3 = highlight_coordinates["P3"]
        P4 = highlight_coordinates["P4"]
        P5 = highlight_coordinates["P5"]
        P6 = highlight_coordinates["P6"]

        # Calcolo EAR (Eye Aspect Ratio)
        numerator: float = dist(P2, P6) + dist(P3, P5)
        denominator: float = 2 * dist(P1, P4)
        EAR: float = numerator / denominator

        return EAR

    @override
    def _actions_timed_FSM(self, frame_data: FrameData, action_ctx: ActionsContext) -> None:
        # TODO: Aggiungere la sottrazione della penalità con il PnP
        # Aggiornamento soglia del frame corrente
        self._current_left_ear_threshold = self._base_left_ear_threshold
        self._current_right_ear_threshold = self._base_right_ear_threshold

        ### Applicazione della Media Mobile Esponenziale (EMA) all'EAR
        # al fine di ridurre il problema del jittering
        ear_ema_coefficient: float = 0.8
        if self._previous_left_ear is not None:
            self._post_ema_sx_ear = (ear_ema_coefficient * self._raw_sx_ear) + (
                (1 - ear_ema_coefficient) * self._previous_left_ear
            )
        else:
            self._post_ema_sx_ear = self._raw_sx_ear
        if self._previous_right_ear is not None:
            self._post_ema_dx_ear = (ear_ema_coefficient * self._raw_dx_ear) + (
                (1 - ear_ema_coefficient) * self._previous_right_ear
            )
        else:
            self._post_ema_dx_ear = self._raw_dx_ear

        ### Filtro di precisione
        left_eye_ratio: float = self._post_ema_sx_ear / self._current_left_ear_threshold
        right_eye_ratio: float = self._post_ema_dx_ear / self._current_right_ear_threshold

        is_left_eye_closed: bool = left_eye_ratio <= 1.0
        is_right_eye_closed: bool = right_eye_ratio <= 1.0
        are_both_eyes_closed: bool = is_left_eye_closed and is_right_eye_closed

        reopening_moment: int | None = None


        current_state: BlinkAction | None = None

        if are_both_eyes_closed:
            if (right_eye_ratio - left_eye_ratio) > self._ear_diff_ratio:
                current_state = BlinkAction.LEFT
            elif (left_eye_ratio - right_eye_ratio) > self._ear_diff_ratio:
                current_state = BlinkAction.RIGHT
            else:
                current_state = BlinkAction.BOTH

        elif current_state is None:
            if is_left_eye_closed:
                current_state = BlinkAction.LEFT
            elif is_right_eye_closed:
                current_state = BlinkAction.RIGHT
            else:
                current_state = None


        if current_state != self._last_state:
            if self._last_state is not None and self._last_state_timestamp_ms is not None:
                reopening_moment = frame_data.timestamp_ms
                blink_time: int = reopening_moment - self._last_state_timestamp_ms

                if (
                    self._min_blink_time_threshold
                    <= blink_time
                    <= self._max_blink_time_threshold
                ):
                    action_ctx.action_registration(
                        state=self._last_state,
                        last_state_timestamp_ms=self._last_state_timestamp_ms,
                        last_state_end_timestamp_ms=reopening_moment
                    )

            self._last_state = current_state
            if current_state is not None:
                self._last_state_timestamp_ms = frame_data.timestamp_ms
            else:
                self._last_state_timestamp_ms = None

        self._previous_left_ear = self._post_ema_sx_ear
        self._previous_right_ear = self._post_ema_dx_ear
