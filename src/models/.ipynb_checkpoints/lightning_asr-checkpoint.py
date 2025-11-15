
import os
import torch    
import math
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from typing import Any, Dict, List, Tuple
from dataclasses import asdict

from .asr_model import ASRModel
from .config import ASRConfig
from src.data.dataset import ASRDataset, MyCollator
from src.utils.metrics import calculate_metrics

class LightningASR(pl.LightningModule):
    """Lightning module for ASR model"""
    
    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        self.model = ASRModel(config)
        self.save_hyperparameters(asdict(config))
        
        # create collator
        self.collator = MyCollator(
            audio_encoder_name=config.audio_encoder_name,
            tokenizer=self.model.tokenizer,
            processor=self.model.audio_encoder.processor
        )
        
        # for accumulating validation/test predictions
        self.all_predictions = []
        self.all_references = []
        self.error_samples: List[Tuple[str, str, float]] = []
        
        self.start_lora = False

    def setup(self, stage: str):
        """Setup datasets"""

        if stage == "fit":
            if self.config.fine_tune_decoder:
                self.model.audio_encoder.dimension_adaptor.train() 
                self.model.audio_encoder.encoder.eval()
                self.model.llm_model.train()
                print('loading new dataset...')
                self.train_dataset = ASRDataset(
                    config=self.config,
                    data_dir=self.config.data_dir,
                    csv_path=self.config.train_csv,
                    tokenizer=self.model.tokenizer,
                    split="train",
                    augment=True  
                )

            else:
                self.model.audio_encoder.dimension_adaptor.train()  
                self.model.audio_encoder.encoder.eval()
                self.model.llm_model.eval()
            
                self.train_dataset = ASRDataset(
                    config=self.config,
                    data_dir=self.config.data_dir,
                    csv_path=self.config.train_csv,
                    tokenizer=self.model.tokenizer,
                    split="train"
                )
            self.val_dataset = ASRDataset(
                config=self.config,
                data_dir=self.config.data_dir,
                csv_path=self.config.val_csv,
                tokenizer=self.model.tokenizer,
                split="val"
            )
            
            # setup warmup steps based on dataset size
            self.config.setup_warmup(len(self.train_dataset))
            
        elif stage == "test":
            self.model.eval()
            self.test_dataset = ASRDataset(
                config=self.config,
                data_dir=self.config.data_dir,
                csv_path=self.config.test_csv,
                tokenizer=self.model.tokenizer,
                split="test"
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-6
        )
        
        from transformers import get_cosine_schedule_with_warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def forward(self, batch):
        """Forward pass"""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        if self.config.fine_tune_decoder:
            self.model.audio_encoder.dimension_adaptor.train() 
            self.model.audio_encoder.encoder.eval()
            self.model.llm_model.train()
        else:
            self.model.audio_encoder.dimension_adaptor.train()  
            self.model.audio_encoder.encoder.eval()
            self.model.llm_model.eval()

        llm_output = self(batch)
        
        if torch.isnan(llm_output.loss) or torch.isinf(llm_output.loss):
            self.log("train/llm_loss_error", 1.0, on_step=True)
            llm_loss = torch.tensor(0.0, device=self.device)
        else:
            llm_loss = llm_output.loss
            
        loss = llm_loss 
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("train/total_loss_error", 1.0, on_step=True)
            loss = torch.tensor(0.0, device=self.device)
            
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        self.model.eval()

        with torch.inference_mode():
            # compute loss
            output = self(batch)
            loss = output.loss

            # generate transcription
            predictions = self.model.transcribe(batch)
            references = self.model.tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True)

            # store predictions and references
            self.all_predictions.extend(predictions)
            self.all_references.extend(references)

            for pred, ref in zip(predictions, references):
                llm_wer = calculate_metrics([pred], [ref])['wer']
                
                if llm_wer > 0:
                    self.error_samples.append((pred, ref, float(llm_wer)))

            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        llm_metrics = calculate_metrics(self.all_predictions, self.all_references)
        
        for name, value in llm_metrics.items():
            self.log(f"val/llm_{name}", value, on_epoch=True)
            
        if len(self.error_samples) > 0:
            sorted_samples = sorted(self.error_samples, key=lambda x: x[2], reverse=True)
            samples_to_log = sorted_samples
            
            error_table = wandb.Table(
                columns=["Prediction", "Reference", "WER"],
                data=[[pred, ref, f"{wer:.4f}"] for pred, ref, wer in samples_to_log]
            )
            self.logger.experiment.log({"val/error_samples": error_table})

        
        self.all_predictions = []
        self.all_references = []
        self.error_samples = []
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        self.model.eval()

        with torch.inference_mode():
            predictions = self.model.transcribe(batch)
            references = self.model.tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True)

            self.all_predictions.extend(predictions)
            self.all_references.extend(references)

            for pred, ref in zip(predictions, references):
                edit_distance = calculate_metrics([pred], [ref])['wer']
                if edit_distance > 0:
                    self.error_samples.append((pred, ref, float(edit_distance)))

        return None
    
    def on_test_epoch_end(self):
        """Called at the end of test"""
        # calculate metrics
        metrics = calculate_metrics(self.all_predictions, self.all_references)
        
        # log metrics
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
            
        if len(self.error_samples) > 0:
            sorted_samples = sorted(self.error_samples, key=lambda x: x[2], reverse=True)
            samples_to_log = sorted_samples
            
            error_table = wandb.Table(
                columns=["Prediction", "Reference", "WER"],
                data=[[pred, ref, f"{wer:.4f}"] for pred, ref, wer in samples_to_log]
            )
            self.logger.experiment.log({"test/error_samples": error_table})
        
        file_path = "predictions_and_references.txt"
        with open(file_path, "a") as file:
            for pred, ref in zip(self.all_predictions, self.all_references):
                file.write(f"Prediction: {pred}\nReference: {ref}\n\n")

        # clear predictions and references
        self.all_predictions = []
        self.all_references = []
        self.error_samples = []
    
    def on_train_epoch_end(self):
        """turn into stage 2 after first epoch"""
        if not self.start_lora:
            print("first epoch completed, now unfreeze LLM model...")
            # unfreeze LLM model parameters in CombinedEmbedding
            self.config.fine_tune_decoder = True
            self.config.use_hint = True
            self.start_lora = True

            self.model.config = self.config
            self.save_hyperparameters(asdict(self.config))

            print(f"Fine tune decoder:{self.model.config.fine_tune_decoder}")
            print(f"Use hint:{self.model.config.use_hint}")

            self.model.enable_lora()
            self.setup("fit")
            
            # print trainable parameters number for confirmation
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"trainable parameters number after unfreezing: {trainable_params:,}")
    
