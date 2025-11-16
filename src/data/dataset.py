import random
import torch
import torchaudio
from torch.utils.data import Dataset
from .audio_augment import AudioAugmentor
import os
import json
import kaldi_io
import numpy as np

class ASRDataset(Dataset):
    """audio dataset"""
    
    def __init__(self, config, data_dir: str, 
                 jsonl_path: str, tokenizer, split: str = "train", augment: bool = False):
        """
        Args:
            config
            data_dir
            jsonl_path: contain audio_path and text
            split: "train", "val", "test"
        """
        self.config = config
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.augment = augment

        self.augmentor = AudioAugmentor(self.config)
        
        # Read JSONL file
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"\nloading {split} dataset:")
        print(f"jsonl_path: {jsonl_path}")
        print(f"data_dir: {data_dir}")
        print(f"original sample number: {len(self.data)}")
        
        self._validate_files()
        print(f"valid sample number: {len(self.data)}\n")
        
        if len(self.data) == 0:
            raise ValueError(f"no valid audio files found! please check data path: {data_dir}")
        
    def _validate_files(self):
        valid_samples = []
        for item in self.data:
            audio_path = item.get('path', '')

            # Check if the path is absolute, otherwise join with data_dir
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.data_dir, audio_path)

            # For .ark files, check if the base file exists
            if 'ark:' in audio_path:
                base_ark_file = audio_path.split(':')[0]
                if os.path.exists(base_ark_file):
                    valid_samples.append(item)
                else:
                    print(f"Warning: .ark file not found - {base_ark_file}")
            # For other formats, check the full path
            else:
                full_audio_path = os.path.normpath(audio_path).replace('\\', '/')
                if os.path.exists(full_audio_path):
                    valid_samples.append(item)
                else:
                    print(f"Warning: audio file not found - {full_audio_path}")
        
        self.data = valid_samples
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int):
        item = self.data[idx]
        
        audio_path = item.get('path', '')

        # Check if the path is absolute, otherwise join with data_dir
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(self.data_dir, audio_path)

        try:
            # Load audio using kaldi-io for .ark files
            if 'ark:' in audio_path:
                # kaldi-io returns a numpy array
                waveform_np = kaldi_io.read_vec_flt(audio_path)
                waveform = torch.from_numpy(waveform_np).float().unsqueeze(0) # Add channel dimension
            else:
                full_audio_path = os.path.normpath(audio_path).replace('\\', '/')
                waveform, sample_rate = torchaudio.load(full_audio_path)

                if sample_rate != self.config.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.config.sampling_rate
                    )
                    waveform = resampler(waveform)

            if self.augment:
                waveform = self.augmentor(waveform)
                
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {str(e)}")
            waveform = torch.zeros((1, int(self.config.max_audio_length * self.config.sampling_rate)))
        
        text = str(item.get('GT', ''))
        
        target_text = text + self.tokenizer.eos_token
    
        words = text.split() 
        num_mask = max(1, int(len(words) * self.config.mask_ratio))
        mask_indices = random.sample(range(len(words)), num_mask if len(words) > num_mask else len(words))

        
        masked_words = words.copy()
        for i in mask_indices:
            masked_words[i] = self.tokenizer.unk_token
            
        hint = ' '.join(masked_words)
      
        prompt_h = (
            "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the speech based on the rough transcript. "
            f"rough transcript:{hint}{self.tokenizer.eos_token}"
            "<|start_header_id|>assistant<|end_header_id|>\n\nAfter understanding the rough transcript and check for grammar errors, the final transcript based on speech are:"
        ) # prompt with hint
        prompt = (
            "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the speech based on the rough transcript. "
            f"rough transcript:None{self.tokenizer.eos_token}" 
            "<|start_header_id|>assistant<|end_header_id|>\n\nAfter understanding the rough transcript and check for grammar errors, the final transcript based on speech are:"
        )
        return waveform, prompt, prompt_h, target_text

class MyCollator:
    def __init__(self, audio_encoder_name, tokenizer, processor):
        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer
        self.hubert_processor = processor

    def __call__(self, batch):
        waveforms = []
        prompts = []
        prompts_h = []
        target_texts = []

        for waveform, prompt, prompt_h, target_text in batch:
            waveforms.append(waveform.squeeze().numpy())  
            prompts.append(prompt)
            prompts_h.append(prompt_h)
            target_texts.append(target_text)
        
        processed = self.hubert_processor(
            waveforms, 
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )
        mel = processed.input_values
        audio_mask = processed.attention_mask
        
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        prompt_ids = prompt_tokens.input_ids
        prompt_mask = prompt_tokens.attention_mask

        prompt_tokens_h = self.tokenizer(
            prompts_h,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        prompt_ids_h = prompt_tokens_h.input_ids
        prompt_mask_h = prompt_tokens_h.attention_mask

        target_tokens = self.tokenizer(
            target_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        target_ids = target_tokens.input_ids
        target_mask = target_tokens.attention_mask

        return {
            'audio_input': mel,
            'audio_mask': audio_mask,
            'prompt_ids': prompt_ids,
            'prompt_ids_h': prompt_ids_h,
            'target_ids': target_ids,
            'prompt_mask': prompt_mask,
            'prompt_mask_h': prompt_mask_h,
            'target_mask': target_mask
        }
