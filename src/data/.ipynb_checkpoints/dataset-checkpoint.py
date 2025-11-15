import random
import torch
import torchaudio
from torch.utils.data import Dataset
from .audio_augment import AudioAugmentor
import pandas as pd
import os

class ASRDataset(Dataset):
    """audio dataset"""
    
    def __init__(self, config, data_dir: str, 
                 csv_path: str, tokenizer, split: str = "train", augment: bool = False):
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
        self.tokenizer = tokenizer
        self.augment = augment

        self.augmentor = AudioAugmentor(self.config)
        
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
        
        text = str(row['text'])
        
        target_text = text + self.tokenizer.eos_token
    
        words = text.split() 
        num_mask = max(1, int(len(words) * self.config.mask_ratio))
        mask_indices = random.sample(range(len(words)), num_mask)
        
        masked_words = words.copy()
        for idx in mask_indices:
            masked_words[idx] = self.tokenizer.unk_token
            
        hint = ' '.join(masked_words)
      
        prompt_h = (
            "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the speech based on the rough transcript. "
            f"rough transcript:{hint}{self.tokenizer.eos_token}"
            "<|start_header_id|>assitant<|end_header_id|>\n\nAfter understanding the rough transcript and check for gramar errors, the final transcript based on speech are:"
        ) # prompt with hint
        prompt = (
            "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the speech based on the rough transcript. "
            f"rough transcript:None{self.tokenizer.eos_token}" 
            "<|start_header_id|>assitant<|end_header_id|>\n\nAfter understanding the rough transcript and check for gramar errors, the final transcript based on speech are:"
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
        
        batch_size = waveforms[0]

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
if __name__ == "__main__":
    # ---------------------- #
    # test config
    # ---------------------- #
    from types import SimpleNamespace
    from transformers import AutoTokenizer

    config = SimpleNamespace(
        sampling_rate=16000,  
        max_audio_length=30,  
        batch_size=3,
        num_workers=32
    )

    test_data_dir = "dataset/LibriSpeech"  
    test_csv_path = "dataset/processed/val.csv" 
    # audio_path,text,topic_desc
    # audio1.wav,"Hello world","greeting"
    # audio2.wav,"How are you","conversation"

    # ---------------------- #
    # inintial setting
    # ---------------------- #
    tokenizer = AutoTokenizer.from_pretrained(
        "llm/Llama-3.2-1B",  
        torch_dtype=torch.bfloat16, 
        padding_side='left'
    )        

    tokenizer.pad_token = tokenizer.bos_token
    try:
        dataset = ASRDataset(config, test_data_dir, test_csv_path, split="train")
        print("\nsample number:", len(dataset))
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}") from e

    collator = MyCollator(
        audio_encoder_name="facebook/hubert-xlarge-ls960-ft",
        tokenizer=tokenizer
    )

    # ---------------------- #
    # Sample test
    # ---------------------- #
    print("\nSample test:")
    sample = dataset[0]
    print(f"waveform tensor shape: {sample[0].shape} (should be [1, sample point])")
    print(f"Prompt : {sample[1]}...") 
    print(f"Target : {sample[3]}...")

    assert sample[0].dim() == 2, f"waveform dimention error，shold be (channels, samples), but got {sample[0].shape}"
    assert isinstance(sample[1], str), "Prompt's type must be str"
    assert isinstance(sample[3], str), "Target's type must be str"

    # ---------------------- #
    # Batch test
    # ---------------------- #
    print("\nBatch test:")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True
    )

    try:
        batch = next(iter(loader))
        print("A batch data generation finished：")
        print(f"audio_input shape: {batch['audio_input'].shape} | dtype: {batch['audio_input'].dtype}")
        print(f"prompt_ids shape: {batch['prompt_ids'].shape} | 示例: {batch['prompt_ids'][0]}")
        print(f"target_ids shape: {batch['target_ids'].shape} | 示例: {batch['target_ids'][0]}")
        
        assert batch['audio_input'].shape[0] == config.batch_size, "Audio batch_size don't match"
        assert batch['prompt_ids'].shape[0] == config.batch_size, "Prompt batch_size don't match"
        assert batch['target_ids'].shape[0] == config.batch_size, "Target batch_size don't match"
        
        assert torch.all(batch['audio_mask'].sum(dim=1) > 0), "存在全零 audio_mask"
        assert torch.all(batch['prompt_mask'].sum(dim=1) > 0), "存在全零 prompt_mask"
        assert torch.all(batch['target_mask'].sum(dim=1) > 0), "存在全零 target_mask"
        
    except Exception as e:
        raise RuntimeError(f"Failed in batch processing: {str(e)}") from e


