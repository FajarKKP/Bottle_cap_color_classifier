# Bottle Cap Color Classifier – bsort

![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-success)
![CI/CD](https://img.shields.io/badge/CI-CD-brightgreen)
[![ci](https://github.com/FajarKKP/Bottle_cap_color_classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/FajarKKP/Bottle_cap_color_classifier/actions/workflows/ci.yml)

## Overview
**bsort** is a complete machine-learning pipeline for classifying bottle cap colors using a YOLO-based model. It provides a Python CLI for training and inference, a YAML configuration system, optional training dependencies, experiment tracking via Weights & Biases, and a Docker-ready deployment pipeline. The project emphasizes reproducibility, automation, clean code, and practical ML engineering.

## Features
- End-to-end ML workflow (training and inference)
- Config-driven pipeline (`config.yaml`)
- YOLO-based detector/classifier using Ultralytics
- Weights & Biases experiment tracking
- Type hints and Google-style docstrings
- CLI entrypoint: `python -m bsort.cli`
- CI pipeline including formatting, linting, tests, and Docker build
- Lightweight runtime image for inference
- Optional training dependencies installed only when needed

## Project Structure
```
root/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── bsort/
│   ├── cli.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
│
├── dataset/
│   ├── train/
│   ├── test/
│   ├── valid/
│   └── data.yaml
│
├── task 1 cap detection/
│   ├── bottle_cap_classifier.ipynb
│   └── bottle_cap_data.ods
│
├── test/
│   └── unit_test.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── settings.yaml
```



## Instalation and setup

Clone the repo
```bash
git clone https://github.com/<your-username>/bsort.git
cd bsort
```

You have the options to do lightweight installation (the minimum dependencies)
```bash
poetry install --only main
```

Or the whole instalation (with ultralytics, etc)
```bash
poetry install --with training
```

## To use the CLI
The bsort can help you in training the model or to do inference

You can adjust the settings for training / inference in root/settings.yaml

## Example of usage
Train
```bash
python -m bsort.cli train --config bsort/config.yaml
```

Inference
```bash
python -m bsort.cli infer --source sample.jpg
```

Help

```bash
python -m bsort.cli --help
```

As a note, you can log in into your wandb account during trainig or inference. It is enabled in the settings.yaml and you can toggle it on or off. 

## Docker Image
There is a docker image you can use related to this project
```bash
docker build -t bsort:latest .
```

## CI/CD Pipeline
This project includes a GitHub Actions workflow file (.github/workflows/ci.yml) that runs linting, tests, and Docker builds automatically 

## Project Results

Datset Overview

Number of images: 27

Classes: 3 (Others, light blue, dark blue)

Train / Valid / Test split: 22 / 3 / 2


Model Overview

Model: Yolov5-nano

Pretrained Weights: COCO

Input image size: 416 (Training) -> 320 (Inference)

Evaluation Metrics

mAP@0.5: 0.82

mAP@0.5-0.95: 0.55

Inference

Inference done using the image from test set

Input image size for inference: 320

Inference Speed: ~25 ms on CPU

CPU Spec used for inference:
| Feature                         | Value                                                             |
| ------------------------------- | ----------------------------------------------------------------- |
| CPU                             | Intel® Xeon® CPU @ 2.20 GHz                                       |
| Physical cores                  | 1                                                                 |
| Logical cores (threads)         | 2 (hyperthreaded)                                                 |
| Cache                           | 56 320 KB (L3)                                                    |
| Max frequency                   | 2.2 GHz                                                           |
| Supported extensions            | SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA, AES, etc. |
| Total RAM (reported separately) | ~12–13 GB in Colab free                                           |

For more detailed experimentation progress, it can be seen here: https://wandb.ai/f-kenichi-kp-none/train

## License
This project is licensed under the MIT License - see the LICENSE file for details.





      
