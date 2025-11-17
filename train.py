import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import random
import numpy as np
from dataclasses import asdict

from src.models.config import ASRConfig
from src.models.lightning_asr import LightningASR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # Load config and update with args
    config = ASRConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    set_seed(42)

    # Logger
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_name,
        entity=config.wandb_entity,
        config=asdict(config),
        offline=True
    )

    # Model
    model = LightningASR(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="{epoch}-{val/llm_wer:.4f}",
        save_top_k=1,
        monitor="val/llm_wer",
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=config.max_grad_norm,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        precision="bf16-mixed" if config.use_amp else "32-true",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
    )

    # Training and testing
    trainer.fit(model)
    trainer.test(model, ckpt_path="best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments based on ASRConfig fields
    for field in asdict(ASRConfig()):
        arg_name = f'--{field}'
        arg_type = type(ASRConfig.__dataclass_fields__[field].default)
        if arg_type == bool:
            parser.add_argument(arg_name, action='store_true')
        else:
            parser.add_argument(arg_name, type=arg_type)

    args = parser.parse_args()
    main(args)
