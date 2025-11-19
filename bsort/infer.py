"""
Inference module for Bottle Cap Color Classifier using Ultralytics YOLO.

Supports toggling WandB logging via the configuration file.
"""

import os

from ultralytics import YOLO

import wandb
from bsort.utils import load_config, setup_wandb


def run_inference(
    config_path: str, image_path: str, save_dir: str = "runs/inference"
) -> None:
    """Run inference on a single image.

    Args:
        config_path: Path to YAML config file.
        image_path: Path to input image.
        save_dir: Directory to save output image with detections.

    Returns:
        None
    """
    cfg = load_config(config_path)
    setup_wandb(cfg)

    # Load YOLO model
    model = YOLO(cfg["weights_path"])

    # Run inference
    results = model.predict(
        source=image_path,
        conf=cfg.get("conf_threshold", 0.25),
        iou=cfg.get("iou_threshold", 0.45),
        save=True,
        save_dir=save_dir,
        verbose=True,
    )

    print(f"Inference completed. Results saved in {os.path.abspath(save_dir)}")

    # WandB logging
    if cfg.get("use_wandb", False):
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # Box coordinates
            confs = boxes.conf.cpu().numpy()  # Box confidences
            cls_ids = boxes.cls.cpu().numpy()  # Class indices

            # Use YOLO's built-in names dict
            class_labels = (
                results[0].names
                if isinstance(results[0].names, dict)
                else dict(enumerate(results[0].names))
            )

            wandb_boxes = {
                "predictions": {
                    "box_data": [
                        {
                            "position": {
                                "minX": float(x1),
                                "minY": float(y1),
                                "maxX": float(x2),
                                "maxY": float(y2),
                            },
                            "class_id": int(cls_id),
                            "score": float(conf),
                        }
                        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids)
                    ],
                    "class_labels": class_labels,
                }
            }

            wandb.log(
                {"predictions": [wandb.Image(results[0].orig_img, boxes=wandb_boxes)]}
            )
        wandb.finish()
