import numpy as np
import torch
import torchaudio
import random

class AudioAugmentor:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, audio):
        """
        Args:
            audio: torch.Tensor, shape [1, T]
        Returns:
            augmented_audio: torch.Tensor, shape [1, T']
        """
        if not self.config.audio_augment:
            return audio
            
        augment_methods = [
            self._add_noise,
            self._change_pitch,
            self._change_speed,
            self._change_volume
        ]
        
        num_augments = random.randint(1, 2)
        selected_methods = random.sample(augment_methods, num_augments)
        
        augmented_audio = audio
        for method in selected_methods:
            augmented_audio = method(augmented_audio)
            
        return augmented_audio
    
    def _add_noise(self, audio):
        noise = torch.randn_like(audio)
        noise_level = random.uniform(0, self.config.noise_factor)
        return audio + noise_level * noise
    
    def _change_pitch(self, audio):
        pitch_shift = random.uniform(-self.config.pitch_shift, self.config.pitch_shift)
        effects = [
            ["pitch", f"{pitch_shift}"],
            ["rate", "16000"]
        ]
        augmented_audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, 16000, effects)
        return augmented_audio
    
    def _change_speed(self, audio):
        speed_factor = random.uniform(1 - self.config.speed_factor, 1 + self.config.speed_factor)
        effects = [["speed", f"{speed_factor}"]]
        augmented_audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, 16000, effects)
        return augmented_audio
    
    def _change_volume(self, audio):
        volume_factor = random.uniform(1 - self.config.volume_factor, 1 + self.config.volume_factor)
        return audio * volume_factor