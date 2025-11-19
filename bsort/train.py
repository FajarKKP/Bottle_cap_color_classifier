"""
Training module for Bottle Cap Color Classifier using Ultralytics YOLO.

Supports toggling WandB logging via the configuration file.
"""

from typing import Any, Dict
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config_path: str) -> None:
    """Train a YOLO model using settings from the YAML config.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        None
    """
    cfg = load_config(config_path)

    # Toggle WandB logging
    use_wandb: bool = cfg.get("use_wandb", True)

    model = YOLO(cfg["model_name"])  # e.g., "yolov8n.pt"

    model.train(
        data=cfg["data_yaml"],              # path to dataset YAML
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        imgsz=cfg["img_size"],
        lr0=cfg["learning_rate"],
        device=cfg.get("device", "cpu"),
        project="runs",
        name=cfg.get("run_name", "bcap_train"),
        pretrained=cfg.get("pretrained", True),
        verbose=True,
        val=True,
        loggers="wandb" if use_wandb else None,
    )

    print("Training completed.")
