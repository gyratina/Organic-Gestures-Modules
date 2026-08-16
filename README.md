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
# 
import time
import logging
from ogm import ActionType, BlinkDetector, CameraConfig

# Optional: enable logging to track internal API events
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Initialize the detector with custom precision parameters
blink_detector = BlinkDetector(
    ear_diff_ratio=0.20,
    calibration_threshold_ratio=0.50,
    sensitivity_coefficient=0.05
)

# Define the callback to handle complex gesture sequences
def process_combinations(actions: tuple[tuple[ActionType, int], ...]) -> None:
    match actions:
        # Both eyes double blink
        case [*_, (ActionType.BOTH, p), (ActionType.BOTH, _)] if p <= 1000:
            print("\nACTION DETECTED: Double Blink!")
            blink_detector.reset_log()

        # Combo: Right eye -> Left eye (max pause 1000ms)
        case [*_, (ActionType.RIGHT, p_dx), (ActionType.LEFT, _)] if p_dx <= 1000:
            print("\nFAST COMBO DETECTED: Right -> Left!")
            blink_detector.reset_log()

        # Combo: Left eye -> Right eye
        case [*_, (ActionType.LEFT, p_sx), (ActionType.RIGHT, _)] if p_sx <= 1000:
            print("\nCOMBO DETECTED: Left -> Right!")
            blink_detector.reset_log()

        # 3-Move Super Combo: Right -> Left -> Right
        case [*_, (ActionType.RIGHT, p1), (ActionType.LEFT, p2), (ActionType.RIGHT, _)] if p1 <= 1000 and p2 <= 1000:
            print("\n3-MOVE SUPER COMBO DETECTED: Right -> Left -> Right!")
            blink_detector.reset_log()

        # Combo: Double Right blink
        case [*_, (ActionType.RIGHT, p), (ActionType.RIGHT, _)] if p <= 1000:
            print("\nCOMBO DETECTED: Double Right!")
            blink_detector.reset_log()

        # Ignore any other sequence
        case _:
            pass

# Define the callback for automatic calibration results
def on_calibration(left_eye: float, right_eye: float) -> None:
    print("\nCalibration finished.")
    print(f"Base Thresholds -> Left: {left_eye:.3f} | Right: {right_eye:.3f}\n")

# Optional: telemetry callback to monitor real-time EAR and Thresholds data
def telemetry(data: dict[str, float]) -> None:
    print(f"EAR_L: {data['sx_eye_ear']:.3f} | EAR_R: {data['dx_eye_ear']:.3f} "
          f"(Threshold L/R: {data['sx_eye_threshold']:.3f}/{data['dx_eye_threshold']:.3f})")

if __name__ == "__main__":
    # Bind callbacks to the detector
    blink_detector.on_blink = process_combinations
    blink_detector.on_calibration = on_calibration
    # blink_detector.telemetry_callback = telemetry  # Uncomment to see live stats
    
    # Configure camera (use 0 for default webcam)
    my_camera = CameraConfig(camera_index=0, fps=60)
    
    # ---------------------------------------------------------
    # 1. Calibration Phase (approx. 3 seconds)
    # ---------------------------------------------------------
    print("Starting Calibration. Please look at the camera with a neutral expression...")
    blink_detector.start(mode="calibrate", camera_config=my_camera)
    
    time.sleep(3.5)
    blink_detector.close()
    
    # ---------------------------------------------------------
    # 2. Gesture Detection Phase
    # ---------------------------------------------------------
    print("Starting Gesture Detection...")
    blink_detector.start(mode="detect", camera_config=my_camera)
    
    try:
        # Keep the main thread alive while the daemon thread does the work
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Safely release resources on exit
        blink_detector.close()
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

#### OGM v0.1.5:
1. Manage horizontal face rotation (when rotating the head, the EAR of the outermost eye spikes up).
2. Perform checks and tests on potential false positives with `BOTH` actions, since the two eyes close and reopen at slightly different milliseconds, in order to evaluate updating the state machine in `precision_filter`.
3. Evaluate the implementation of a virtual camera for the video channel to feed into MediaPipe.

#### OGM v0.2.0:
1. Brainstorm how to restructure the API architecture to support the creation of other facial modules.
2. Rewrite the API architecture.
3. Add the eyebrows module.
4. Add the `GEMINI.md`, `CLAUDE.md`, and `AGENTS.md` files as symlinks to the `.rules` file with precise instructions on the style and direction to adopt for development with the contribution of AI agents.

#### Future versions:
1. Mouth movement gestures module. (v0.3.0)
2. Eye movement gestures module. (v0.4.0)
3. Hand movement gestures module. (v0.5.0)
4. Rewriting the core API in C++ and Rust. (v1.0.0?)
5. Too early to think this forward.

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
