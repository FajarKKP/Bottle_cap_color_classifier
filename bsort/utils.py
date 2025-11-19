"""
Utility functions for Bottle Cap Color Classifier.
"""

from typing import Dict


def get_color_mapping() -> Dict[int, str]:
    """Return a mapping from class IDs to color names.

    Example:
        0 → light blue
        1 → dark blue
        2 → others

    Returns:
        dict: {class_id: color_name}
    """
    return {
        0: "light_blue",
        1: "dark_blue",
        2: "others",
    }


def map_prediction_to_color(class_id: int) -> str:
    """Map YOLO class ID to color string.

    Args:
        class_id: int, class ID predicted by YOLO

    Returns:
        str: color name
    """
    mapping = get_color_mapping()
    return mapping.get(class_id, "unknown")
