import torch.nn as nn
from quantizer.env import AttrDict
from quantizer.models import Generator, Quantizer, Encoder
import torch
import json
import os

class Vocoder(nn.Module):
    def __init__(self, config_path=None, ckpt_path=None, with_encoder=False, 
                 use_xcodec=False, xcodec_model_name="facebook/xcodec-base",
                 n_code_groups=4, sample_rate=16000):
        """
        Vocoder supporting both original MQTTS quantizer and XCodec.
        
        Args:
            config_path: Path to original quantizer config (if use_xcodec=False)
            ckpt_path: Path to original quantizer checkpoint (if use_xcodec=False)
            with_encoder: Whether to include encoder (original quantizer only)
            use_xcodec: Whether to use XCodec instead of original quantizer
            xcodec_model_name: HuggingFace model name for XCodec
            n_code_groups: Number of code groups
            sample_rate: Audio sample rate
        """
        super(Vocoder, self).__init__()
        self.use_xcodec = use_xcodec
        
        if use_xcodec:
            # Use XCodec-based vocoder
            # NOTE: XCodec weights are loaded from HuggingFace (pre-trained)
            # They are NOT trained locally. To fine-tune XCodec, set freeze_encoder=False
            # and ensure vocoder parameters are not frozen in trainer.py
            from modules.vocoder_xcodec import VocoderXCodec
            # Default: freeze XCodec (like original vocoder)
            # Can be changed later via freeze_encoder attribute
            self.vocoder_xcodec = VocoderXCodec(
                xcodec_model_name=xcodec_model_name,
                n_code_groups=n_code_groups,
                sample_rate=sample_rate,
                freeze_encoder=True  # Freeze by default (can be changed for fine-tuning)
            )
            self.quantizer = None
            self.generator = None
            self.encoder = None
        else:
            # Original MQTTS quantizer
            if config_path is None or ckpt_path is None:
                raise ValueError("config_path and ckpt_path required when use_xcodec=False")
            
            ckpt = torch.load(ckpt_path)
            with open(config_path) as f:
                data = f.read()
            json_config = json.loads(data)
            self.h = AttrDict(json_config)
            self.quantizer = Quantizer(self.h)
            self.generator = Generator(self.h)
            self.generator.load_state_dict(ckpt['generator'])
            self.quantizer.load_state_dict(ckpt['quantizer'])
            if with_encoder:
                self.encoder = Encoder(self.h)
                self.encoder.load_state_dict(ckpt['encoder'])
            self.vocoder_xcodec = None

    def forward(self, x, spkr):
        """Decode codes to audio."""
        if self.use_xcodec:
            return self.vocoder_xcodec(x, spkr)
        else:
            return self.generator(self.quantizer.embed(x), spkr)

    def encode(self, x):
        """Encode audio to codes."""
        if self.use_xcodec:
            return self.vocoder_xcodec.encode(x)
        else:
            batch_size = x.size(0)
            c = self.encoder(x.unsqueeze(1))
            q, loss_q, c = self.quantizer(c)
            c = [code.reshape(batch_size, -1) for code in c]
            return torch.stack(c, -1) #N, T, 4
