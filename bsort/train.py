"""
Training module for Bottle Cap Color Classifier using Ultralytics YOLO.

Supports toggling WandB logging via the configuration file and prompts login if needed.
"""

from typing import Any, Dict

import yaml
from ultralytics import YOLO

import wandb


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
    if cfg.get("use_wandb", False):
        try:
            # Check if already logged in
            wandb.require("service")
        except wandb.errors.UsageError:
            print("You are not logged into WandB. Please follow instructions to login:")
            wandb.login()  # prompts user in terminal

        wandb.init(
            project=cfg.get("wandb_project", "bsort_project"),
            name=cfg.get("wandb_run_name", "bsort_run"),
            config=cfg,
            reinit=True,
        )

    model = YOLO(cfg["model_name"])  # e.g., "yolov8n.pt"

    model.train(
        data=cfg["data_yaml"],  # path to dataset YAML
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        imgsz=cfg["img_size"],
        lr0=cfg["learning_rate"],
        device=cfg.get("device", "cpu"),
        project="runs",
        name=cfg.get("run_name", "bsort_train"),
        pretrained=cfg.get("pretrained", True),
        verbose=True,
        val=True,
    )

    print("Training completed.")

    # Finish WandB run if used
    if cfg.get("use_wandb", False):
        wandb.finish()
