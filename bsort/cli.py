"""
Command-line interface for Bottle Cap Color Classifier.
Supports training and inference.
"""

import argparse
from bsort.train import run_training
from bsort.infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Bottle Cap Color Classifier CLI")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands: train or infer")

    # ------------------ Train ------------------ #
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config file"
    )

    # ------------------ Infer ------------------ #
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config file"
    )
    infer_parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to input image"
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.config)
    elif args.command == "infer":
        results = run_inference(
            model_path=r"D:\Passion\Inventor\Code\Bottle_cap_color_classifier\model\yolov8nano_8200.pt",  # adjust if needed
            image_path=args.image
        )
        print("Inference results:", results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
