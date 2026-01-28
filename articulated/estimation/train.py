"""Training script for state estimation models.

Usage:
    python -m articulated.estimation.train --config articulated/configs/estimation/rnn.yaml
"""

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from articulated.estimation.datamodule import EstimationDataModule
from articulated.estimation.model import StateEstimationModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    """Train a state estimation model."""
    # Set seed
    if "seed" in config:
        L.seed_everything(config["seed"])

    # Initialize data module
    data_config = config.get("data", {})
    datamodule = EstimationDataModule(**data_config)

    # Initialize model
    model_config = config.get("model", {})
    model = StateEstimationModel(**model_config)

    # Setup logging
    logging_config = config.get("logging", {})
    logger = None
    if logging_config.get("wandb", False):
        logger = WandbLogger(
            project=logging_config.get("project", "articulated-estimation"),
            name=logging_config.get("name"),
            save_dir=logging_config.get("save_dir", "logs"),
        )

    # Setup callbacks
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/estimation"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer_config = config.get("trainer", {})
    trainer = L.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", 1),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
    )

    # Train
    trainer.fit(model, datamodule)

    print(f"\nTraining complete! Best checkpoint saved to: {checkpoint_dir}")
    print("Team RL can load this model using:")
    print(f"  StateEstimationModel.load_for_embedding('{checkpoint_dir}/best.ckpt')")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train state estimation model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
