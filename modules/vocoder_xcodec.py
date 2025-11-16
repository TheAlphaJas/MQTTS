"""
XCodec-based vocoder for MQTTS.
Replaces the original Generator + Quantizer vocoder with XCodec.
"""
import torch.nn as nn
import torch
import json
from modules.xcodec_wrapper import XCodecWrapper


class VocoderXCodec(nn.Module):
    """
    Vocoder using XCodec for encoding and decoding.
    Compatible with MQTTS pipeline.
    """
    def __init__(self, xcodec_model_name: str = "facebook/xcodec-base", 
                 n_code_groups: int = 4, sample_rate: int = 16000,
                 freeze_encoder: bool = False):
        """
        Args:
            xcodec_model_name: HuggingFace model identifier
            n_code_groups: Number of code groups (must match MQTTS config)
            sample_rate: Audio sample rate
            freeze_encoder: Whether to freeze XCodec weights
        """
        super(VocoderXCodec, self).__init__()
        self.n_code_groups = n_code_groups
        self.sample_rate = sample_rate
        
        # Initialize XCodec wrapper
        self.xcodec = XCodecWrapper(
            model_name=xcodec_model_name,
            n_code_groups=n_code_groups,
            sample_rate=sample_rate,
            freeze_encoder=freeze_encoder
        )
        
        # For speaker conditioning, we'll add a speaker projection layer
        # XCodec doesn't have built-in speaker conditioning, so we add it
        self.spkr_linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512)
        )
    
    def forward(self, codes: torch.Tensor, spkr: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to audio waveform.
        
        Args:
            codes: Code indices of shape (B, T, n_code_groups)
            spkr: Speaker embedding of shape (B, 512)
            
        Returns:
            Audio waveform of shape (B, T_audio)
        """
        # XCodec doesn't directly support speaker conditioning
        # We can condition by modifying the embeddings, but for now
        # we'll decode directly and add speaker conditioning later if needed
        audio = self.xcodec.decode(codes)
        
        # Apply speaker conditioning if needed (optional)
        # This is a placeholder - actual implementation depends on requirements
        spkr_proj = self.spkr_linear(spkr)
        
        return audio
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete codes.
        
        Args:
            audio: Audio tensor of shape (B, T) or (B, 1, T)
            
        Returns:
            Codes of shape (B, T', n_code_groups)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
        
        # Encode with XCodec
        z_q, loss, codes_list = self.xcodec(audio)
        
        # Convert codes list to tensor format
        batch_size = audio.size(0)
        # Get temporal dimension from codes
        T = codes_list[0].size(0) // batch_size if codes_list[0].size(0) > batch_size else 1
        
        # Reshape codes
        codes = torch.stack(codes_list, dim=-1)  # (B*T', n_code_groups)
        codes = codes.reshape(batch_size, T, self.n_code_groups)
        
        return codes

