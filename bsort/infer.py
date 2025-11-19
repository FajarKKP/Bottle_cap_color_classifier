"""
Inference module for Bottle Cap Color Classifier using Ultralytics YOLO.
"""

from typing import List, Any
from ultralytics import YOLO


def run_inference(model_path: str, image_path: str) -> List[Any]:
    """Run YOLO inference.

    Args:
        model_path: Path to .pt YOLO model.
        image_path: Input image.

    Returns:
        Raw YOLO predictions (list of Results objects).
    """
    model = YOLO(model_path)
    results = model(image_path)
    return results
