import torch
import torchaudio
from transformers import AutoProcessor, AutoFeatureExtractor
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Dict, Optional

class Dataset(Dataset):
    """audio dataset"""
    
    def __init__(self, config, data_dir: str, 
                 csv_path: str, split: str = "train"):
        """
        Args:
            config
            data_dir
            csv_path: contain audio_path and text
            split: "train", "val", "test"
        """
        self.config = config
        self.data_dir = data_dir
        self.split = split
        
        self.data = pd.read_csv(csv_path)
        print(f"\nloading {split} dataset:")
        print(f"csv_path: {csv_path}")
        print(f"data_dir: {data_dir}")
        print(f"original sample number: {len(self.data)}")
        
        self.data['audio_path'] = self.data['audio_path'].apply(
            lambda x: os.path.normpath(x).replace('\\', '/')
        )
        
        self._validate_files()
        print(f"valid sample number: {len(self.data)}\n")
        
        if len(self.data) == 0:
            raise ValueError(f"no valid audio files found! please check data path: {data_dir}")
        
    def _validate_files(self):
        valid_samples = []
        for idx, row in self.data.iterrows():
            audio_path = os.path.join(self.data_dir, row['audio_path'])
            audio_path = os.path.normpath(audio_path).replace('\\', '/')
            if os.path.exists(audio_path):
                valid_samples.append(row)
            else:
                print(f"Warning: data do not exist - {audio_path}")
        
        self.data = pd.DataFrame(valid_samples)
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        
        audio_path = os.path.join(self.data_dir, row['audio_path'])
        audio_path = os.path.normpath(audio_path).replace('\\', '/')
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {str(e)}")
            waveform = torch.zeros((1, int(self.config.max_audio_length * self.config.sampling_rate)))
        
        text = str(row['text'])
        topic_desc = str(row['topic_desc']) if 'topic_desc' in row else ""
        
        return waveform, text, topic_desc

class ASRDataset(Dataset):
    """ASR dataset"""
    
    def __init__(self, config, data_dir: str, 
                 csv_path: str, split: str = "train"):
        super().__init__(config, data_dir, csv_path, split)
    
    def __getitem__(self, idx: int):
        waveform, text, topic_desc = super().__getitem__(idx)

        ctc_label = [self.config.ctc_vocab.index(char) if char in self.config.ctc_vocab 
                      else self.config.ctc_blank_id for char in text]

        

        return waveform, ctc_label
        
class MyCollator:
    def __init__(self, audio_encoder_name):
        self.audio_encoder_name = audio_encoder_name
        self.hubert_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    def __call__(self, batch):
        waveforms = []
        ctc_labels = []

        max_length = max(waveform.size(-1) for waveform, _ in batch)
        
        for waveform, ctc_label in batch:
            if waveform.size(-1) < max_length:
                padding = torch.zeros(waveform.size(0), max_length - waveform.size(-1))
                waveform = torch.cat([waveform, padding], dim=1)        
            waveform = waveform.float()
            
            ctc_labels.append(torch.tensor(ctc_label))
            waveforms.append(waveform.squeeze().numpy())  
        
        mel = self.hubert_processor(
            waveforms, 
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        ).input_values
        
        ctc_labels = torch.nn.utils.rnn.pad_sequence(
            ctc_labels, 
            batch_first=True, 
            padding_value=0
        )
        
        return mel, ctc_labels