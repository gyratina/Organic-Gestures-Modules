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
from ogm import ActionType, BlinkDetector, CameraConfig

# Initialize the detector - For greater accuracy, you should adjust these 3 parameters.
#                           See the DocStrings for more info about them (i'm lazy).
blink_detector = BlinkDetector(
    ear_diff=0.05, calibration_threshold_ratio=0.60, sensitivity_coefficient=0.15
)

# Define the callback for automatic calibration
def on_calibration(left_eye: float, right_eye: float):
    blink_detector.left_ear_threshold = left_eye
    blink_detector.right_ear_threshold = right_eye
    print(f"Calibration finished.\nLeft EAR: {left_eye}, Right EAR: {right_eye}")

# Define the callback to handle gesture sequences
def on_actions(actions: list[tuple[ActionType, int]]):
    match actions:
        # Single blink of the left eye
        case [(ActionType.LEFT, _)]:
            print("Action: LEFT BLINK")
            blink_detector.reset_log()

        # Single blink of both eyes
        case [(ActionType.BOTH, _)]:
            print("Action: BOTH EYES BLINK")
            blink_detector.reset_log()

        # Combo example: Right eye + Left eye (max pause 800ms)
        case [*_, (ActionType.RIGHT, p), (ActionType.LEFT, _)] if p <= 800:
            print(f"Action: RIGHT -> LEFT (pause: {p}ms)")
            blink_detector.reset_log()

        # Wait state: The user blinked the right eye, waiting for combo
        case [*_, (ActionType.RIGHT, _)]:
            print("Waiting for complete combo...")
            pass

        # Ignore any other sequence
        case _:
            pass

if __name__ == "__main__":
    # Bind callbacks
    blink_detector.on_blink = on_actions
    blink_detector.on_calibration_callback = on_calibration
    
    # Configure camera (use 0 for default webcam)
    my_camera = CameraConfig()
    
    # Calibration Phase
    print("Starting Calibration. Please look at the camera with a neutral expression for 3 seconds...")
    blink_detector.start(mode="calibrate", camera_config=my_camera)
    
    # Wait for the background thread to finish the 3-second calibration
    time.sleep(3.5)
    
    # Safely close the calibration thread and release resources
    blink_detector.close()
    
    # Gesture Detection Phase
    print("Starting Gesture Detection...")
    blink_detector.start(mode="detect", camera_config=my_camera)
    
    try:
        # Keep the main thread alive while the background daemon thread does the work.
        # ---> YOU CAN RUN YOUR OWN APPLICATION LOOP OR GUI HERE <---
        while True:    # This cycle is only for testing the API
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Always remember to safely release resources on exit
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
1. Blinking gestures module <-- **Major features implemented!**
3. Eyebrow movement gestures module. (v0.2.0)
4. Eye movement gestures module. (v0.3.0)
5. Mouth movement gestures module. (v0.4.0)
2. Hand movement gestures module. (v0.5.0)
5. Rewriting the core API in C++ and Rust. (v1.0.0)
6. Creating bindings for JavaScript.
7. Creating other bindings?

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
