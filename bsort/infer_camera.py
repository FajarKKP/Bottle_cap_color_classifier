"""
Real-time Bottle Cap Color Classifier using Ultralytics YOLO.
Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO

from bsort.utils import load_config


def run_camera_inference(config_path: str):
    """
    Run real-time inference on laptop camera feed.

    Args:
        config_path: Path to settings.yaml.
        conf: Confidence threshold for predictions.
        camera_index: Camera device index (0, 1, etc.)
    """
    cfg = load_config(config_path)
    camera_index = cfg["camera_index"]
    conf = cfg["confidence_threshold"]

    print("Loading YOLO model from settings.yaml")
    model = YOLO(cfg["model_path"])
    print("Model loaded successfully.")

    print(f"Opening camera (index={camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}. Try changing camera_index.")
        return
    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        # Predict directly on frame
        results = model.predict(source=frame, conf=conf, verbose=False)

        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

        cv2.imshow("Bottle Cap Color Classifier", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. All windows closed.")
