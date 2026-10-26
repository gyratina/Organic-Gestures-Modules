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
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from math import dist
from queue import Queue
from typing import ClassVar, override

import cv2 as cv
import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks import python as py
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)

from ..modules_base import Module
from ..ogm import FrameData

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class BlinksModule(Module):
    MODULE_NAME: ClassVar[str] = "blinks"

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
        max_combo_delay: int = 2000,
        ear_diff_ratio: float = 0.20,
        calibration_threshold_ratio: float = 0.50
    ) -> None:
        self.left_eye_coordinates: dict[str, tuple[float, float]] = {}
        self.right_eye_coordinates: dict[str, tuple[float, float]] = {}

        # Valori EAR grezzi del frame corrente
        self._raw_sx_ear: float = 0.0
        self._raw_dx_ear: float = 0.0

        # Soglie di apertura dell'occhio
        self._base_left_ear_threshold: float = base_left_ear_threshold
        self._base_right_ear_threshold: float = base_right_ear_threshold
        self._calibration_threshold_ratio: float = calibration_threshold_ratio

        # Variabili di conteggio e somma per la calibrazione degli occhi
        self._sum_left_ear: float = 0.0
        self._sum_right_ear: float = 0.0
        self._count_ear: int = 0

    @override
    def _frame_processing(self, frame_data: FrameData) -> None:
        self._get_coordinates(
            face_landmarks=frame_data.landmarks,
            frame_height=frame_data.frame_height,
            frame_width=frame_data.frame_width,
            highlight_points=self._LEFT_EYE,
            output_dict=self.left_eye_coordinates
        )
        self._get_coordinates(
            face_landmarks=frame_data.landmarks,
            frame_height=frame_data.frame_height,
            frame_width=frame_data.frame_width,
            highlight_points=self._RIGHT_EYE,
            output_dict=self.right_eye_coordinates
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
        output_dict: dict[str, tuple[float, float]]
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
    def _calcs(
        self,
        highlight_coordinates: dict[str, tuple[float, float]]
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
