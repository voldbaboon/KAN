import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .encoder import AudioEncoder
from peft import get_peft_model, LoraConfig, TaskType

class ASRModel(nn.Module):
    """a ASR model, consist of audio encoder and llm model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"Fine tune decoder:{self.config.fine_tune_decoder}")
        print(f"Use hint:{self.config.use_hint}")
        
        # initialize the audio encoder
        self.audio_encoder = AudioEncoder(config)
        for param in self.audio_encoder.encoder.parameters():
            param.requires_grad = False
        
        # initialize the language model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_name, torch_dtype=torch.bfloat16)

        if self.config.fine_tune_decoder:
            self.enable_lora()
        else:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.config.pad_token_id
            print(f"pad token:{self.tokenizer.pad_token}, pad token id:{self.tokenizer.pad_token_id}")

        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token_id = self.config.pad_token_id
            print(f"pad token:{self.tokenizer.unk_token}, pad token id:{self.tokenizer.unk_token_id}")

    def enable_lora(self):
        # Integrate LoRA for decoder fine-tuning if enabled
        inference_mode = not self.training
        if inference_mode == False:
            print("LoRA training enabled...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference_mode,
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        
        if self.training:   
            for name, param in self.llm_model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    def gradient_checkpointing_enable(self):
        """enable gradient checkpointing"""
        self.audio_encoder.gradient_checkpointing_enable()
        self.llm_model.gradient_checkpointing_enable()
        
    def gradient_checkpointing_disable(self):
        """disable gradient checkpointing"""
        self.audio_encoder.gradient_checkpointing_disable()
        self.llm_model.gradient_checkpointing_disable()

    def featurefusion(self, speech_embeds, prompt_embeds):
        """
        feature fusion
        """
        
        combined_embeds = torch.cat([speech_embeds , prompt_embeds], dim=1)
        
        return combined_embeds
        
    def encode(self, speech, mel_mask, prompt_ids, prompt_mask, target_ids = None, target_mask = None):
        """
        encode audio and prompt words
        Args:
            speech: audio input
            prompts: pre-processed prompt words
            target_texts: output prompt words (reference text)
        Returns:
            combined_embeds: combined embeddings
            atts: attention mask
            label_ids: label ids for loss computation
            ctc_output: ctc output
        """
        batch_size = speech.size(0)
    
        speech_embeds, ctc_output, audio_mask = self.audio_encoder(speech, mel_mask)
            
        if not self.training and self.config.use_hint:
            ctc_hint = self.audio_encoder.ctc_decode(ctc_output)  
            prompts = []
            for hint in ctc_hint:
                
                prompt = (
                    "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the speech based on the rough transcript. "
                    f"rough transcript:{hint}{self.tokenizer.eos_token}"
                    "<|start_header_id|>assistant<|end_header_id|>\n\nAfter understanding the rough transcript and check for grammar errors, the final transcript based on speech are:"
                )
                prompts.append(prompt)

            tokens = self.tokenizer(
                prompts,
                padding=True,
                return_tensors='pt',
                add_special_tokens=False
            )
            prompt_ids = tokens.input_ids.to(self.config.device)
            prompt_mask = tokens.attention_mask.to(self.config.device)
        
        if target_ids is not None and target_mask is not None:           
            ids = torch.cat([prompt_ids, target_ids], dim=1)
            text_mask = torch.cat([prompt_mask, target_mask], dim=1) # for loss compution
        else:
            ids = prompt_ids
            text_mask = prompt_mask # for generation
        
        embedder = self.llm_model.get_input_embeddings()

        prompt_embeds = embedder(ids)

        # combine all features
        combined_embeds = self.featurefusion(speech_embeds, prompt_embeds)
        
        # create label, set non-target parts to -100
        input_token_length = speech_embeds.shape[1] + prompt_ids.shape[1]

        atts = torch.cat([
            audio_mask,
            text_mask
        ], 1).to(torch.int64)
        
        if target_ids is not None and target_mask is not None: 
            label_ids = torch.cat([
                torch.ones([batch_size, input_token_length], device=combined_embeds.device) * -100,
                target_ids * target_mask + (-100) * (1 - target_mask)  
            ], 1).to(torch.int64)
            return combined_embeds, atts, label_ids
        else:
            return combined_embeds, atts
            
        
    def forward(self, batch):
        """
        forward propagation
        Args:
            batch: dictionary containing:
                - audio_input: audio input tensor
                - prompt_ids: pre-processed prompt ids
                - target_ids: target text ids
                - attention_mask: combined attention mask
        Returns:
            output: model output
        """
        audio_input = batch['audio_input']
        mel_mask = batch['audio_mask']
        target_ids = batch['target_ids']
        target_mask = batch['target_mask']
        if self.config.use_hint:
            prompt_ids = batch['prompt_ids_h']
            prompt_mask = batch['prompt_mask_h']
        else:
            prompt_ids = batch['prompt_ids']
            prompt_mask = batch['prompt_mask']
        
        # encode input
        combined_embeds, atts, label_ids = self.encode(
            audio_input,
            mel_mask,
            prompt_ids, 
            prompt_mask,
            target_ids,
            target_mask
        )

        llm_output = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return llm_output
            
    def transcribe(self, batch):
        """
        transcribe audio to text
        Args:
            batch: dictionary containing:
                - audio_input: audio input tensor
                - prompt_ids: pre-processed prompt ids
                - mask: attention mask
        Returns:
            transcription: transcribed text
        """
        # generate transcription
        with torch.inference_mode():
            audio_input = batch['audio_input']
            mel_mask = batch['audio_mask']
            prompt_ids = batch['prompt_ids']
            prompt_mask = batch['prompt_mask']

                
            # encode input
            combined_embeds, atts = self.encode(
                audio_input, 
                mel_mask,
                prompt_ids, 
                prompt_mask,
            )
            
            outputs = self.llm_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=atts,
                num_beams=5,
                max_length=512,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            llm_generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        return llm_generated_text
    

   
