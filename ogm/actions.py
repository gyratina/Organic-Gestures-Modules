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
# file-name: actions.py
# ========================================================================


import threading
from dataclasses import dataclass
from enum import Enum
from queue import Queue


class ActionType(Enum):
    pass

class BlinkAction(ActionType):
    LEFT = 0
    RIGHT = 1
    BOTH = 2


@dataclass(slots=True)
class ActionsContext:
    actions_list: list[tuple[ActionType, int]]
    actions_queue: Queue[tuple[tuple[ActionType, int], ...] | None]
    rlock: threading.RLock

    max_combo_delay: int = 2000
    previous_state_timestamp_ms: int | None = None

    def _clear(self) -> None:
        """
        Clears the logged action sequence buffer and resets inter-blink combo timers.
        Typically invoked after matching a desired combo.

        Acquires the internal reentrant lock to ensure thread-safe clearing of `ActionsContext.actions_queue`
        and resetting of `ActionsContext.previous_state_timestamp_ms`.
        """
        with self.rlock:
            self.actions_list.clear()
            self.previous_state_timestamp_ms = None

    def action_registration(
        self,
        state: ActionType,
        last_state_timestamp_ms: int,
        last_state_end_timestamp_ms: int
    ) -> None:
        with self.rlock:
            if not self.actions_list and self.previous_state_timestamp_ms is None:
                self.actions_list.append((state, 0))
                self.previous_state_timestamp_ms = last_state_end_timestamp_ms

            elif self.previous_state_timestamp_ms is not None:
                lapse = (
                    last_state_timestamp_ms
                    - self.previous_state_timestamp_ms
                )
                if lapse > self.max_combo_delay:
                    self.actions_list.clear()
                    self.actions_list.append((state, 0))
                    self.previous_state_timestamp_ms = last_state_end_timestamp_ms
                else:
                    self.actions_list[-1] = (self.actions_list[-1][0], lapse)
                    self.actions_list.append((state, 0))
                    self.previous_state_timestamp_ms = last_state_end_timestamp_ms

            # Creazione copia (snapshot) immutabile (tuple) della lista actions_list
            snapshot: tuple[tuple[ActionType, int], ...] = tuple(
                self.actions_list
            )

        # Passaggio dello snapshot di actions_list alla coda thread-safe che chiama la
        # callback su un thread separato
        self.actions_queue.put(item=snapshot)
