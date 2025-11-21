import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from torch.utils import data
from modules.vocoder import Vocoder
import soundfile as sf
import librosa
from librosa.util import normalize
from pyannote.audio import Inference
from modules.style_encoder import StyleEncoder
from modules.text_encoder import SemanticEncoder
import random
from tqdm import tqdm

class Wav2TTS_infer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.hp.init = 'std'
        self.TTSdecoder = TTSDecoder(hp, len(self.hp.phoneset))
        
        # Updated dimension for spkr_linear (Style + Pyannote)
        self.spkr_embed_dim = 512
        self.style_encoder = None
        # Check config/args for style encoder usage. Assuming inference uses same config as training.
        if hasattr(hp, 'style_encoder_type') and hp.style_encoder_type == 'style_tts2':
            self.style_encoder = StyleEncoder()
            self.spkr_embed_dim = 128 + 512

        self.spkr_linear = nn.Linear(self.spkr_embed_dim, hp.hidden_size)
        self.phone_embedding = nn.Embedding(len(self.hp.phoneset), hp.hidden_size, padding_idx=self.hp.phoneset.index('<pad>'))
        
        self.semantic_encoder = SemanticEncoder(device='cuda')
        
        self.load()
        self.spkr_embedding = Inference("pyannote/embedding", window="whole")
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path, with_encoder=True)
        
        # Adapter for Vocoder
        self.style_to_vocoder = None
        if self.style_encoder:
            self.style_to_vocoder = nn.Linear(128 + 512, 512)

        # Reload style_to_vocoder from checkpoint if it exists
        state_dict = torch.load(self.hp.model_path)['state_dict']
        if self.style_to_vocoder and 'style_to_vocoder.weight' in state_dict:
            print("Loading style_to_vocoder from checkpoint")
            self.style_to_vocoder.load_state_dict({
                'weight': state_dict['style_to_vocoder.weight'],
                'bias': state_dict['style_to_vocoder.bias']
            })

    def load(self):
        state_dict = torch.load(self.hp.model_path)['state_dict']
        model_dict = self.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    print(f"Skipping loading parameter {k} due to shape mismatch. Checkpoint: {v.shape}, Model: {model_dict[k].shape}")
                    continue
                new_state_dict[k] = v
        print(self.load_state_dict(new_state_dict, strict=False))

    def forward(self, wavs, phones, texts):
        self.eval()
        with torch.no_grad():
            batch_size = len(wavs)
            
            # 1. Speaker Embeddings
            speaker_embeddings = []
            style_embeddings = []
            
            for wav in wavs:
                if self.hp.spkr_embedding_path:
                     # Assuming spkr_embedding_path is for Pyannote
                     # Logic for style encoder inference from file?
                     # If we use style encoder, we need raw audio.
                     # wavs contains raw audio (loaded in infer.py) or paths.
                     # infer.py loads audio if spkr_embedding_path is None.
                     pass
                else:
                     # wav is numpy array of audio
                     w = normalize(wav) * 0.95
                     w_torch = torch.FloatTensor(w).unsqueeze(0)
                     
                     # Pyannote
                     spk_emb = self.spkr_embedding({'waveform': w_torch, 'sample_rate': self.hp.sample_rate})
                     speaker_embeddings.append(spk_emb)
                     
                     # Style Encoder
                     if self.style_encoder:
                         style_emb = self.style_encoder(w_torch.cuda())
                         style_embeddings.append(style_emb)

            speaker_embeddings = torch.cuda.FloatTensor(np.array(speaker_embeddings))
            norm_spkr = F.normalize(speaker_embeddings, dim=-1)
            
            if self.style_encoder:
                style_embeddings = torch.cat(style_embeddings, dim=0)
                norm_style = F.normalize(style_embeddings, dim=-1)
                combined_spkr = torch.cat([norm_style, norm_spkr], dim=-1)
                
                # Project for Transformer
                speaker_embedding_proj = self.spkr_linear(combined_spkr)
                
                # Project for Vocoder
                voc_spkr = combined_spkr
                if self.style_to_vocoder:
                    voc_spkr = self.style_to_vocoder(combined_spkr)
            else:
                speaker_embedding_proj = self.spkr_linear(norm_spkr)
                voc_spkr = norm_spkr
            
            # 2. Semantic Embeddings
            semantic_embeddings = self.semantic_encoder(texts)
            
            low_background_noise = torch.randn(batch_size, int(self.hp.sample_rate * 5.0)) * self.hp.prior_noise_level
            base_prior = self.vocoder.encode(low_background_noise.cuda())
            if self.hp.clean_speech_prior:
                prior = base_prior[:, :self.hp.prior_frame]
            else:
                prior = None
            phone_features, phone_masks = [], []
            for phone in phones:
                phone = [self.hp.phoneset.index(ph) for ph in phone if ph != ' ' and ph in self.hp.phoneset]
                phone = np.array(phone)
                phone_features.append(phone)
            #Pad phones
            maxlen = max([len(x) for x in phone_features])
            for i, ph in enumerate(phone_features):
                to_pad = maxlen - len(ph)
                pad = np.zeros([to_pad,], dtype=np.float32)
                pad.fill(self.hp.phoneset.index('<pad>'))
                phone_features[i] = np.concatenate([ph, pad], 0)
                mask = [False] * len(ph)+ [True] * to_pad
                phone_masks.append(mask)
            phone_masks = torch.cuda.BoolTensor(phone_masks)
            phone_features = torch.cuda.LongTensor(phone_features)
            phone_features = self.phone_embedding(phone_features)
            
            # Pass semantic embeddings to inference
            synthetic = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, speaker_embedding_proj, semantic_embeddings, phone_masks, prior=prior)
            
            padded_synthetic, lengths = [], []
            maxlen = max([len(x) for x in synthetic])
            for i, s in enumerate(synthetic):
                to_pad = maxlen - len(s)
                lengths.append(len(s) * 256) # Have to change according to vocoder stride!
                pad = base_prior[i, base_prior.size(1)//2].unsqueeze(0).expand(to_pad, -1)
                if self.hp.clean_speech_prior:
                    s = torch.cat([prior[i, :], s, pad], 0)
                else:
                    s = torch.cat([s, pad], 0)
                padded_synthetic.append(s)
            padded_synthetic = torch.stack(padded_synthetic, 0)
            
            # Use correct speaker embedding for vocoder
            synthetic = self.vocoder(padded_synthetic, voc_spkr)
            outputs = []
            for l, s in zip(lengths, synthetic):
                if self.hp.clean_speech_prior:
                    l = l + self.hp.prior_frame * 256
                outputs.append(s[0, : l].cpu().numpy())
            return outputs
