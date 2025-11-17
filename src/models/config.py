from dataclasses import dataclass, field
from typing import Optional, List
import torch
  
@dataclass
class ASRConfig:
    
    # model config
    audio_enc_dim: int = 1280
    llm_dim: int = 3072
    audio_processor_name: str = "/aistor/sjtu/hpc_stor01/home/wangchenghao/models/facebook/hubert-xlarge-ls960-ft"
    audio_encoder_name: str = "/aistor/sjtu/hpc_stor01/home/wangchenghao/models/facebook/hubert-xlarge-ls960-ft"
    llm_model_name: str = "/aistor/sjtu/hpc_stor01/home/wangchenghao/models/LLM-Research/Llama-3.1-8B-Instruct"
    
    # training stage 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 4
    warmup_steps: int = 1500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    pad_token_id:int = 128004
    
    # training stage 2
    mask_ratio: float = 0.35
    fine_tune_decoder: bool = True
    use_hint: bool = True
    lora_learning_rate: float = 1e-4
    rank: int = 16
    alpha: int = 32
    
    # data config
    max_audio_length: int = 30  # seconds
    sampling_rate: int = 16000
    num_workers: int = 8
    dataset_size: int = None  # will be set when loading dataset
       
    # augment config
    noise_factor: float = 0.1     
    pitch_shift:float = 2        
    speed_factor:float = 0.2     
    volume_factor:float = 0.1    
    
    # optimization config
    use_amp: bool = True  # use automatic mixed precision
    empty_cache_freq: int = 100  # clear GPU cache frequency
    checkpoint_activation: bool = True  # use gradient checkpointing
    
    # wandb config
    wandb_project: str = "llm_asr"
    wandb_name: str = None  # will be set automatically if None
    wandb_entity: str = None
    
    # paths
    data_dir: str = "dataset/LibriSpeech"
    train_jsonl: str = "dataset/processed/train.jsonl"
    val_jsonl: str = "dataset/processed/val.jsonl"
    test_jsonl: str = "dataset/processed/test.jsonl"
    output_dir: str = "checkpoints"
    
    # device
    device: str = None
    
    def setup_warmup(self, dataset_size: int):
        """
        Args:
            dataset_size
        """
        self.dataset_size = dataset_size
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        steps_per_epoch = dataset_size / (self.batch_size * num_gpus * self.gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * self.max_epochs
        
        print("\nTraining schedule:")
    
        print(f"Effective batch size: {self.batch_size * num_gpus * self.gradient_accumulation_steps}")
        print(f"Steps per epoch: {steps_per_epoch:.1f}")
        print(f"Total training steps: {self.total_steps:.1f}")
        print(f"Warmup steps: {self.warmup_steps:,}")
        print("\nLearning rate schedule:")
        print(f"Initial learning rate: 0")
        print(f"Learning rate after warmup: {self.learning_rate:.2e}")
        print(f"Decay type: Cosine annealing")
        print(f"Minimum learning rate: 0")
    
    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        # set wandb name if not provided
        if self.wandb_name is None:
            self.wandb_name = f"asr_{self.llm_model_name.split('/')[-1]}_{self.max_epochs}ep_bs{self.batch_size}"
            