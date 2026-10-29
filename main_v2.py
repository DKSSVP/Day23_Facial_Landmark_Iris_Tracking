
import cv2
import mediapipe as mp
import time
from pathlib import Path
from datetime import datetime


MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"
PROJECT_FOLDER = Path(__file__).resolve().parent
ABSENCE_TIMEOUT_SECONDS = 10

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe iris landmark indices.
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


def draw_point(frame, x, y, color, radius=2):
    """Draw a point if its coordinates are inside the image."""
    height, width = frame.shape[:2]

    if 0 <= x < width and 0 <= y < height:
        cv2.circle(frame, (x, y), radius, color, -1)


def main():
    if not MODEL_PATH.is_file():
        print(f"Error: Model file not found: {MODEL_PATH}")
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Unable to access the webcam.")
        return

    window_name = "Day 23 - Face Landmark and Iris Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.perf_counter()
    previous_time = start_time
    fps = 0.0
    last_timestamp_ms = -1

    # Track continuous absence of a detected face.
    absence_started_at = None
    screenshot_saved_for_absence = False

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print("MediaPipe Face Landmarker loaded successfully.")
            print("Webcam started. Press q or ESC to exit.")
            print(
                "A screenshot will be saved after "
                "10 continuous seconds without a detected face."
            )

            while True:
                success, frame = camera.read()

                if not success:
                    print("Error: Unable to read webcam frame.")
                    break

                frame_count += 1

                # Mirror the preview.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms, last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                capture_screenshot = False
                absence_duration = 0.0

                if result.face_landmarks:
                    # A face has returned: reset the absence timer.
                    absence_started_at = None
                    screenshot_saved_for_absence = False

                    landmarks = result.face_landmarks[0]

                    # Draw facial landmarks.
                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        draw_point(frame, x, y, (0, 255, 0), 1)

                    iris_groups = [
                        (
                            "Left iris",
                            LEFT_IRIS_INDICES,
                            (255, 0, 255),
                        ),
                        (
                            "Right iris",
                            RIGHT_IRIS_INDICES,
                            (0, 165, 255),
                        ),
                    ]

                    for label, indices, color in iris_groups:
                        valid_points = [
                            landmarks[i]
                            for i in indices
                            if i < len(landmarks)
                        ]

                        if not valid_points:
                            continue

                        pixel_points = []

                        for point in valid_points:
                            px = int(point.x * width)
                            py = int(point.y * height)
                            draw_point(frame, px, py, color, 3)
                            pixel_points.append((px, py))

                        center_x = sum(
                            point[0] for point in pixel_points
                        ) // len(pixel_points)

                        center_y = sum(
                            point[1] for point in pixel_points
                        ) // len(pixel_points)

                        cv2.circle(
                            frame,
                            (center_x, center_y),
                            5,
                            color,
                            2,
                        )

                        text_y = 90 if label == "Left iris" else 120

                        cv2.putText(
                            frame,
                            f"{label}: ({center_x}, {center_y})",
                            (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                else:
                    # Start the timer on the first frame without a face.
                    if absence_started_at is None:
                        absence_started_at = time.perf_counter()

                    absence_duration = (
                        time.perf_counter() - absence_started_at
                    )

                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                    remaining = max(
                        0,
                        ABSENCE_TIMEOUT_SECONDS - absence_duration,
                    )

                    cv2.putText(
                        frame,
                        f"Screenshot in: {remaining:.1f} seconds",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                    if (
                        absence_duration >= ABSENCE_TIMEOUT_SECONDS
                        and not screenshot_saved_for_absence
                    ):
                        capture_screenshot = True

                current_time = time.perf_counter()
                elapsed = current_time - previous_time

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = (
                        instant_fps
                        if fps == 0
                        else 0.8 * fps + 0.2 * instant_fps
                    )

                previous_time = current_time

                cv2.putText(
                    frame,
                    "Day 23: Facial Landmark and Iris Tracking",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Frames: {frame_count} | FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Save the current annotated frame in the project folder.
                if capture_screenshot:
                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    screenshot_path = PROJECT_FOLDER / (
                        f"no_face_detected_{timestamp}.png"
                    )

                    cv2.putText(
                        frame,
                        "10-second absence detected - screenshot saved",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 255),
                        2,
                    )

                    saved = cv2.imwrite(
                        str(screenshot_path), frame
                    )

                    if saved:
                        screenshot_saved_for_absence = True
                        print(
                            "No-face screenshot saved:",
                            screenshot_path,
                        )
                    else:
                        print(
                            "Error: Unable to save screenshot:",
                            screenshot_path,
                        )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("Exit key detected.")
                    break

                try:
                    if cv2.getWindowProperty(
                        window_name, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except cv2.error:
                    break

    except Exception as exc:
        print(f"Application error: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")
        print("Webcam released and application closed.")


if __name__ == "__main__":
    main()
