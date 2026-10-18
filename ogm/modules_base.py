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
# file-name: modules_base.py
# ========================================================================

from abc import ABC, abstractmethod

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from .ogm import FrameData


class Module(ABC):
    @abstractmethod
    def _frame_processing(self, frame_data: FrameData) -> None:
        pass

    @abstractmethod
    def _get_coordinates(
        self,
        face_landmarks: list[NormalizedLandmark],
        frame_height: int,
        frame_width: int,
        highlight_points: dict[str, int],
        output_dict: dict[str, tuple[float, float]]
    ) -> None:
        pass

    @abstractmethod
    def _calcs(
        self,
        highlight_coordinates: dict[str, tuple[float, float]]
    ) -> float:
        pass
