"""
XCodec wrapper module for MQTTS integration.
Replaces the original Encoder + Quantizer with XCodec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XCodecModel, XCodecConfig
from typing import List, Tuple, Optional


class XCodecWrapper(nn.Module):
    """
    Wrapper for XCodec that provides interface compatible with MQTTS quantizer.
    
    XCodec uses Residual Vector Quantization (RVQ) with multiple codebooks.
    This wrapper maps XCodec's code structure to MQTTS's expected format.
    """
    def __init__(self, model_name: str = "facebook/xcodec-base", n_code_groups: int = 4, 
                 sample_rate: int = 16000, freeze_encoder: bool = False):
        """
        Args:
            model_name: HuggingFace model identifier for XCodec
            n_code_groups: Number of code groups (must match MQTTS config)
            sample_rate: Audio sample rate
            freeze_encoder: Whether to freeze XCodec encoder weights
        """
        super().__init__()
        self.n_code_groups = n_code_groups
        self.sample_rate = sample_rate
        
        # Load pretrained XCodec model
        print(f"Loading XCodec model: {model_name}")
        try:
            self.xcodec_model = XCodecModel.from_pretrained(model_name)
            self.config = self.xcodec_model.config
        except Exception as e:
            print(f"Error loading XCodec model: {e}")
            print("Trying to load with AutoModel...")
            from transformers import AutoModel
            self.xcodec_model = AutoModel.from_pretrained(model_name)
            self.config = self.xcodec_model.config
        
        # XCodec typically uses multiple RVQ codebooks
        # We need to map these to n_code_groups
        # XCodec's codebooks are in the quantizer
        self.n_codebooks = getattr(self.config, 'num_codebooks', 
                                   getattr(self.config, 'num_quantizers', 8))  # Default XCodec codebooks
        
        # Ensure we have enough codebooks to map to n_code_groups
        if self.n_codebooks < n_code_groups:
            print(f"Warning: XCodec has {self.n_codebooks} codebooks but need {n_code_groups} groups. "
                  f"Will use first {n_code_groups} codebooks.")
        
        # Get codebook sizes from config
        self.n_codes_per_book = getattr(self.config, 'codebook_size', 
                                        getattr(self.config, 'vocab_size', 1024))
        
        if freeze_encoder:
            # Freeze XCodec parameters (only fine-tune if needed)
            for param in self.xcodec_model.parameters():
                param.requires_grad = False
        
        print(f"XCodec initialized: {self.n_codebooks} codebooks, {self.n_codes_per_book} codes per book")
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to continuous features (before quantization).
        
        Args:
            audio: Audio tensor of shape (B, 1, T) or (B, T)
            
        Returns:
            Continuous features of shape (B, C, T')
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, T) -> (B, 1, T)
        
        # XCodec expects (batch_size, sequence_length) for audio
        if audio.size(1) == 1:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
        
        # Encode with XCodec
        outputs = self.xcodec_model.encode(audio, return_dict=True)
        
        # XCodec returns audio_codes and audio_scales
        # We need the continuous features before quantization
        # For now, we'll use the encoded features directly
        # The actual encoding happens in the forward method
        return outputs.audio_codes
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Encode and quantize audio, returning quantized features and codes.
        
        Args:
            audio: Audio tensor of shape (B, 1, T) or (B, T)
            
        Returns:
            z_q: Quantized features (B, C, T')
            loss: Quantization loss (scalar)
            codes: List of code indices for each codebook (B*T', n_code_groups)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, T) -> (B, 1, T)
        
        # XCodec expects (batch_size, sequence_length)
        if audio.size(1) == 1:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
        
        batch_size = audio.size(0)
        
        # Encode and quantize with XCodec
        # XCodec API: model(input_values) returns codes
        try:
            outputs = self.xcodec_model(audio, return_dict=True)
            # Try different possible output formats
            if hasattr(outputs, 'audio_codes'):
                audio_codes = outputs.audio_codes  # (B, T', n_codebooks)
            elif hasattr(outputs, 'codes'):
                audio_codes = outputs.codes
            elif hasattr(outputs, 'codebook_indices'):
                audio_codes = outputs.codebook_indices
            else:
                # Fallback: try to get from last_hidden_state or similar
                raise AttributeError("Could not find codes in XCodec output")
            
            # Get quantized values if available
            if hasattr(outputs, 'audio_values'):
                audio_values = outputs.audio_values
            elif hasattr(outputs, 'quantized'):
                audio_values = outputs.quantized
            else:
                audio_values = None
        except Exception as e:
            print(f"Error in XCodec forward: {e}")
            print("Attempting alternative encoding method...")
            # Fallback: use encode method if available
            if hasattr(self.xcodec_model, 'encode'):
                audio_codes = self.xcodec_model.encode(audio)
            else:
                raise RuntimeError(f"Could not encode with XCodec: {e}")
            audio_values = None
        
        # Map XCodec codebooks to n_code_groups
        # Take first n_code_groups codebooks
        if audio_codes.size(-1) > self.n_code_groups:
            audio_codes = audio_codes[..., :self.n_code_groups]
        elif audio_codes.size(-1) < self.n_code_groups:
            # Pad with zeros if needed (shouldn't happen with proper config)
            padding = torch.zeros(
                audio_codes.size(0), audio_codes.size(1), 
                self.n_code_groups - audio_codes.size(-1),
                device=audio_codes.device, dtype=audio_codes.dtype
            )
            audio_codes = torch.cat([audio_codes, padding], dim=-1)
        
        # Reshape codes to match MQTTS format: (B*T', n_code_groups)
        B, T, G = audio_codes.shape
        codes_flat = audio_codes.reshape(B * T, G)  # (B*T', n_code_groups)
        codes_list = [codes_flat[:, i] for i in range(G)]
        
        # Get quantized features
        # XCodec's audio_values is already quantized, but we need to reshape
        # For compatibility, we'll use the quantized embeddings
        z_q = self.embed(codes_flat.unsqueeze(0))  # (1, B*T', C)
        z_q = z_q.squeeze(0).transpose(0, 1)  # (C, B*T')
        z_q = z_q.reshape(-1, B, T).transpose(1, 2)  # (C, B, T') -> (B, C, T')
        
        # Compute quantization loss (commitment loss)
        # XCodec handles this internally, but we compute for compatibility
        loss = torch.tensor(0.0, device=audio.device)  # XCodec loss is handled internally
        
        return z_q, loss, codes_list
    
    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert code indices to embeddings.
        
        Args:
            codes: Code indices of shape (B, T, n_code_groups) or (1, B*T, n_code_groups)
            
        Returns:
            Embedded features of shape (B, C, T) or (1, B*T, C)
        """
        original_shape = codes.shape
        if codes.dim() == 3:
            B, T, G = codes.shape
            codes = codes.reshape(B * T, G)
        elif codes.dim() == 2:
            # Already flattened
            pass
        
        # Get embeddings from XCodec quantizer
        # XCodec stores codebooks in the quantizer
        embeddings = []
        for i in range(min(self.n_code_groups, self.n_codebooks)):
            codebook = self.xcodec_model.quantizer.codebooks[i]  # (codebook_size, dim)
            code_indices = codes[:, i]  # (B*T,)
            # Clamp indices to valid range
            code_indices = torch.clamp(code_indices, 0, codebook.size(0) - 1)
            emb = codebook[code_indices]  # (B*T, dim)
            embeddings.append(emb)
        
        # Pad if needed
        if self.n_code_groups > self.n_codebooks:
            dim = embeddings[0].size(-1)
            for i in range(self.n_codebooks, self.n_code_groups):
                padding = torch.zeros(codes.size(0), dim, device=codes.device, dtype=embeddings[0].dtype)
                embeddings.append(padding)
        
        # Concatenate embeddings
        z_q = torch.cat(embeddings, dim=-1)  # (B*T, C_total)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            B, T, G = original_shape
            z_q = z_q.reshape(B, T, -1)
        
        return z_q
    
    def decode(self, codes: torch.Tensor, audio_length: Optional[int] = None) -> torch.Tensor:
        """
        Decode codes back to audio using XCodec decoder.
        
        Args:
            codes: Code indices of shape (B, T, n_code_groups)
            audio_length: Optional target audio length
            
        Returns:
            Decoded audio of shape (B, T_audio)
        """
        # Ensure codes have the right number of codebooks
        if codes.size(-1) < self.n_codebooks:
            # Pad codes to match XCodec's expected number of codebooks
            padding = torch.zeros(
                codes.size(0), codes.size(1), 
                self.n_codebooks - codes.size(-1),
                device=codes.device, dtype=codes.dtype
            )
            codes = torch.cat([codes, padding], dim=-1)
        elif codes.size(-1) > self.n_codebooks:
            # Truncate to match
            codes = codes[..., :self.n_codebooks]
        
        # Decode with XCodec
        # XCodec decoder expects codes in specific format
        try:
            # Try different decode methods
            if hasattr(self.xcodec_model, 'decode'):
                audio = self.xcodec_model.decode(codes, audio_scales=None)
            elif hasattr(self.xcodec_model, 'generate'):
                # Some models use generate instead of decode
                audio = self.xcodec_model.generate(codes)
            else:
                raise AttributeError("No decode method found")
        except Exception as e:
            # Fallback: use embeddings and decode
            print(f"Warning: Direct decode failed: {e}. Using embedding-based decode.")
            # Get quantized features
            z_q = self.embed(codes)  # (B, T, C)
            # Reshape for decoder
            z_q = z_q.transpose(1, 2)  # (B, C, T)
            # Use XCodec's decoder directly if available
            if hasattr(self.xcodec_model, 'decoder'):
                audio = self.xcodec_model.decoder(z_q)
            elif hasattr(self.xcodec_model, 'decode_from_embeddings'):
                audio = self.xcodec_model.decode_from_embeddings(z_q)
            else:
                # Fallback: return zeros (should not happen with proper XCodec)
                print(f"Warning: Could not decode. Returning zeros.")
                T_audio = codes.size(1) * 320  # Approximate frame-to-sample ratio
                audio = torch.zeros(codes.size(0), T_audio, device=codes.device)
        
        return audio

