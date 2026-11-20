# Organic Gestures Modules (OGM)

**OGM** is a Python API that allows the implementation of highly customizable hands and facial gestures.

Currently, OGM supports only blinking gestures. These are fully customizable, allowing the API user to build potentially infinite combinations of actions (RAM permitting).

---

## Installation
If you download this repository, navigate inside and run:

```bash
pip install .
```

Alternatively you can install the last stable release:
```bash
pip install ogm-vision
```


---

## Usage

> [!WARNING]
> **OGM IS IN ACTIVE DEVELOPMENT**
> 
> It is highly likely that the way to do certain things with OGM might change frequently across versions, as the library is still in early development.
> To get an idea of what's coming, check out the roadmap below.

I have included DocStrings within the API files, so if any information is missing here, you should still have access to everything you need right in your IDE.

***Usage Example (Calibration & Detection):***
```python
### file_name: main.py

import logging
import threading
from time import sleep

from ogm import ActionType, BlinkAction, BlinksModule, CameraConfig, OGM

# Optional: enable logging to track internal API events
logging.basicConfig(
    level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
)

# 1. Initialize the gesture modules you want to enable
blinks = BlinksModule(
    ear_diff_ratio=0.20,
    calibration_threshold_ratio=0.50,
)

# 2. Initialize the OGM orchestrator with the selected modules
app = OGM(modules=[blinks])

# Event used to wait until the 3-second calibration phase completes
calib_done = threading.Event()


# Define the callback to handle gesture sequences (and cross-module combos)
def process_combinations(actions: tuple[tuple[ActionType, int], ...]) -> None:
    match actions:
        # Double blink with both eyes (max pause 1000 ms)
        case [*_, (BlinkAction.BOTH, p), (BlinkAction.BOTH, _)] if p <= 1000:
            print("\nCOMBO DETECTED: Double Blink!")
            app.clear()  # Clear the combo buffer after a match

        # Combo: Right eye -> Left eye (max pause 1000 ms)
        case [*_, (BlinkAction.RIGHT, p_dx), (BlinkAction.LEFT, _)] if p_dx <= 1000:
            print("\nCOMBO DETECTED: Right -> Left!")
            app.clear()

        # Combo: Left eye -> Right eye
        case [*_, (BlinkAction.LEFT, p_sx), (BlinkAction.RIGHT, _)] if p_sx <= 1000:
            print("\nCOMBO DETECTED: Left -> Right!")
            app.clear()

        # 3-Move Super Combo: Right -> Left -> Right
        case [
            *_,
            (BlinkAction.RIGHT, p1),
            (BlinkAction.LEFT, p2),
            (BlinkAction.RIGHT, _),
        ] if p1 <= 1000 and p2 <= 1000:
            print("\nCOMBO DETECTED: Right -> Left -> Right!")
            app.clear()

        # Combo: Double Right blink
        case [*_, (BlinkAction.RIGHT, p), (BlinkAction.RIGHT, _)] if p <= 1000:
            print("\nCOMBO DETECTED: Double Right!")
            app.clear()

        # Ignore any other sequence until a valid combo matches
        case _:
            pass


# Define the callback for automatic calibration results (keyed by MODULE_NAME)
def on_calibration(calibration_data: dict[str, dict[str, float]]) -> None:
    blink_data = calibration_data.get(BlinksModule.MODULE_NAME, {})
    print("\nCalibration finished.")
    print(
        f"Base Thresholds -> Left: {blink_data.get('left_ear_threshold', 0.0):.3f} | "
        f"Right: {blink_data.get('right_ear_threshold', 0.0):.3f}\n"
    )
    calib_done.set()  # Unblock the main thread so we can move to detection


if __name__ == "__main__":
    # Bind callbacks to the OGM orchestrator
    app.on_calibration = on_calibration
    app.on_action = process_combinations

    # Configure camera (use 0 for the default webcam)
    my_camera = CameraConfig(camera_index=0, fps=60)

    # ---------------------------------------------------------
    # 1. Calibration Phase (approx. 3 seconds)
    # ---------------------------------------------------------
    print("Starting Calibration. Please look at the camera with a neutral expression...")
    app.start(mode="calibrate", camera_config=my_camera)

    calib_done.wait()  # Wait for on_calibration to signal completion
    app.close()        # Release resources before restarting in detect mode

    # ---------------------------------------------------------
    # 2. Gesture Detection Phase
    # ---------------------------------------------------------
    print("Starting Gesture Detection...")
    app.start(mode="detect", camera_config=my_camera)

    try:
        # Keep the main thread alive while background daemon threads run
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Safely stop engines/sentinel threads and release the webcam
        app.close()
```

---

## Acknowledgments & Legal
This library is released under the **[Apache License 2.0](LICENSE)**.

OGM is built as a wrapper and mathematical layer on top of [Google MediaPipe](https://developers.google.com/mediapipe) for high-performance, real-time facial and hand landmark detection.

The OGM library bundles the `face_landmarker.task` model, which is provided by Google LLC under the **Apache License 2.0**. 
For full copyright notices and third-party attribution, please refer to the **[NOTICE](NOTICE)** and **[LICENSE](LICENSE)** files, or visit the official [MediaPipe repository](https://github.com/google-ai-edge/mediapipe).

---

## Development Roadmap
The roadmap I have set for the development of this API is:

#### OGM v0.2.0:
1. Brainstorm how to restructure the API architecture to support the creation of other facial modules.
2. Rewrite the API architecture.

#### OGM v0.2.1
- Add the feature of disabling or enabling single modules both at runtime and not without closing their engine thread.

#### OGM v0.2.2:
- Rework entirely the current face orientation calculations with the implementation of the PnP algorithm (Perspective-n-Point).
- Add local telemetry (for tests only).

#### OGM v0.2.3:
- Perform checks and tests on potential false positives with `BOTH` actions, since the two eyes close and reopen at slightly different milliseconds, in order to evaluate updating the state machine in `BlinksModule._actions_timed_FSM`, and in case: update the FSM.

#### OGM v0.3.0:
1. Add the eyebrows module.
2. *Evaluate the implementation of a virtual camera for the video channel to feed into MediaPipe.

#### OGM v0.3.1:
1. Add saving functionality for calibration data.
2. Allow modules to be launched at runtime, provided they have been previously calibrated.

#### OGM v0.4.0:
- Mouth movement gestures module.

#### OGM v0.5.0:
- Eye movement gestures module.

#### OGM v0.6.0:
- Hand movement gestures module.

#### OGM v1.0.0:
- Unlock Python GIL and tests.
- Add the `GEMINI.md`, `CLAUDE.md`, and `AGENTS.md` files as symlinks to a `.rules` file with precise instructions on the style and direction to adopt for development with the contribution of AI agents.

#### Future versions:
- Rewriting the core API in C++ and Rust (?)

---

```
###
# project: Organic Gestures Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina (on GitHub)
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# licence: Apache Licence 2.0
###
```

**To learn more about me, check <a href="https://valerioditommaso.dev/en">my website</a>.**
