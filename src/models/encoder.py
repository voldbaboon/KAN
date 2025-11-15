import torch
import torch.nn as nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoConfig, HubertForCTC, AutoProcessor
 
class dimension_adaptor(nn.Module):
    """Downsample by 4 and project into LLM dimension"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.CNN_block = nn.Sequential(
            nn.GELU(),
            
            nn.Conv1d(config.audio_enc_dim, config.llm_dim // 2, kernel_size=4, stride=2),
            nn.BatchNorm1d(config.llm_dim // 2),
            nn.GELU(),
            
            nn.Conv1d(config.llm_dim // 2, config.llm_dim, kernel_size=4, stride=2),
            nn.BatchNorm1d(config.llm_dim),
            nn.GELU(),  

            nn.Conv1d(config.llm_dim , config.llm_dim, kernel_size=4, stride=1),
        )

    def forward(self, features):

        features = features.transpose(1, 2)
        features = self.CNN_block(features)
        features = features.transpose(1, 2)
        
        return features

class AudioEncoder(nn.Module):
    """Audio encoder, using HUBERT to extract features and adjust dimensions through CNN"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # print(f"\nAudioEncoder Configuration:")
        # print(f"Audio encoding dimension (audio_enc_dim): {config.audio_enc_dim}")
        # print(f"LLM dimension (llm_dim): {config.llm_dim}")
        
        # load HUBERT encoder
        self.encoder = HubertForCTC.from_pretrained(config.audio_encoder_name)
        self.processor = AutoProcessor.from_pretrained(config.audio_processor_name)
        
        # adaptor layer with downsampling and Layernorm
        self.dimension_adaptor = dimension_adaptor(self.config)

    def forward(self, audio_input, mask):
        """
        forward propagation
        Args:
            audio_input: audio input of shape [batch_size, sequence_length]
        Returns:
            features: shape [batch_size, max_length, llm_dim]
            attention_mask: shape [batch_size, max_length]
            ctc_output: CTC logits
        """
        if audio_input.dim() == 3:
            # if input is [batch_size, channels, sequence_length]
            audio_input = audio_input.squeeze(1)
        elif audio_input.dim() == 4:
            # if input is [batch_size, channels, 1, sequence_length]
            audio_input = audio_input.squeeze(1).squeeze(1)
            
        # get HUBERT features
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_values=audio_input,
                output_hidden_states=True,
                attention_mask=mask
            )

            features = encoder_outputs.hidden_states[-1] # [B, T, D]
            ctc_output = encoder_outputs.logits  
            
        # adjust dimensions through CNN adaptor layer
        features = self.dimension_adaptor(features)  # [B, T//4, D_new]
        
        batch_size, seq_len, _ = features.shape
            
        # create attention mask for padded features
        atts = torch.ones((batch_size, seq_len), dtype=torch.long, device=features.device)
        
        return features, ctc_output, atts
    
    def ctc_decode(self, ctc_output):
        """
        CTC decode
        Args:
            ctc_output: shape [batch_size, sequence_length, vocab_size]
        Returns:
            list of decoded sequences
        """
        
        predictions = torch.argmax(ctc_output, dim=-1)  # [batch_size, sequence_length]
        decoded_sequences = self.processor.batch_decode(predictions)
  
        return decoded_sequences
    
    def gradient_checkpointing_enable(self):
        """enable gradient checkpointing"""
        self.encoder.gradient_checkpointing_enable()
        
    def gradient_checkpointing_disable(self):
        """disable gradient checkpointing"""
        self.encoder.gradient_checkpointing_disable() 

if __name__ == "__main__":
    from transformers import AutoProcessor
    import torch
    import torchaudio

    processor = AutoProcessor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    audio_input_paths = [
        "dataset/LibriSpeech/test-clean/61/70968/61-70968-0000.flac",
        "dataset/LibriSpeech/test-clean/61/70968/61-70968-0001.flac"
    ]

    audio_inputs = []
    for path in audio_input_paths:
        audio_input, _ = torchaudio.load(path)
        audio_input = audio_input.squeeze().numpy() 
        print(f"loading {path}, shape: {audio_input.shape}")
        audio_inputs.append(audio_input)

    processed = processor(
        audio_inputs,
        sampling_rate=16000,
        padding=True,          
        return_tensors="pt",
    )

    mel_spec = processed.input_values
    attention_mask = processed.attention_mask

    print("Correct mel_spec shape:", mel_spec.shape)        
    print("Correct attention_mask shape:", attention_mask.shape)  



    
    
    

