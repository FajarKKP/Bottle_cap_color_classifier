"""
Command-line interface for Bottle Cap Color Classifier (bsort)
"""

import argparse

from bsort.infer import run_inference
from bsort.train import run_training
from bsort.infer_camera import run_camera_inference


def main():
    """CLI entry point for running training/inference commands."""

    parser = argparse.ArgumentParser(
        description="BSort: Bottle Cap Color Classifier CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- Train -------- #
    train_parser = subparsers.add_parser("train", help="Train the YOLO model")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to settings YAML file"
    )

    # -------- Inference -------- #
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument(
        "--config", type=str, required=True, help="Path to settings YAML file"
    )
    infer_parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    infer_parser.add_argument(
        "--save_dir",
        type=str,
        default="runs/inference",
        help="Directory to save output",
    )

    # -------- Camera Inference -------- #
    camera_parser = subparsers.add_parser(
        "infer_with_camera", help="Run real-time inference using built-in cameera"
    )
    camera_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to settings YAML File where infer will use the model",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.config)
    elif args.command == "infer":
        run_inference(args.config, args.image, args.save_dir)
    elif args.command == "infer_with_camera":
        run_camera_inference(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
