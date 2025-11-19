"""
Utility functions for Bottle Cap Color Classifier.
"""

import yaml

import wandb


def load_config(path: str):
    """Load a YAML configuration file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_wandb(cfg: dict):
    """Initialize WandB if enabled in config."""
    if not cfg.get("use_wandb", False):
        return

    try:
        wandb.require("service")
    except wandb.errors.UsageError:
        print("You are not logged into WandB. Please follow instructions to login:")
        wandb.login()

    wandb.init(
        project=cfg.get("wandb_project", "bsort_project"),
        name=cfg.get("wandb_run_name", "bsort_run"),
        config=cfg,
        reinit=True,
    )
