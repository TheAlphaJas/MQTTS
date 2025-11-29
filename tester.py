import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from torch.utils import data
from modules.vocoder import Vocoder
from modules.style_encoder import StyleEncoder
import soundfile as sf
import librosa
from librosa.util import normalize
from pyannote.audio import Inference
import random
from tqdm import tqdm

class Wav2TTS_infer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.hp.init = 'std'
        self.TTSdecoder = TTSDecoder(hp, len(self.hp.phoneset))
        
        # First, check the checkpoint to see if style encoder was used during training
        checkpoint = torch.load(self.hp.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # CRITICAL: Check if checkpoint was trained with semantic encoder
        has_semantic = any('semantic' in k.lower() for k in state_dict.keys())
        if has_semantic:
            print("\n" + "="*80)
            print("[ERROR] This checkpoint was trained with SEMANTIC encoder!")
            print("[ERROR] You are using vanilla tester.py (no semantic support).")
            print("[ERROR] Please use tester_semantic.py and infer_semantic.py instead!")
            print("="*80 + "\n")
            raise ValueError("Checkpoint/Tester mismatch: Use tester_semantic.py for semantic checkpoints")
        
        has_style_encoder = any('style_encoder' in k for k in state_dict.keys())
        has_style_to_vocoder = any('style_to_vocoder' in k for k in state_dict.keys())
        
        # Determine speaker embedding dimension based on checkpoint
        if has_style_encoder or has_style_to_vocoder:
            print("[INFO] Checkpoint was trained WITH style encoder. Initializing style encoder...")
            spkr_embed_dim = 128 + 512  # StyleTTS2 (128) + Pyannote (512)
            self.style_encoder = StyleEncoder()
            self.style_to_vocoder = nn.Linear(128 + 512, 512)
        else:
            print("[INFO] Checkpoint was trained WITHOUT style encoder. Using Pyannote only...")
            spkr_embed_dim = 512
            self.style_encoder = None
            self.style_to_vocoder = None

        self.spkr_linear = nn.Linear(spkr_embed_dim, hp.hidden_size)
        self.phone_embedding = nn.Embedding(len(self.hp.phoneset), hp.hidden_size, padding_idx=self.hp.phoneset.index('<pad>'))

        # Load checkpoint weights
        self.load()
        
        # Initialize Pyannote speaker embedding model
        self.spkr_embedding = Inference("pyannote/embedding", window="whole")
        
        # Initialize vocoder from checkpoint (not random weights)
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path, with_encoder=True)
        
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

    def forward(self, wavs, phones):
        self.eval()
        with torch.no_grad():
            batch_size = len(wavs)
            speaker_embeddings = []
            style_embeddings = []
            
            for wav in wavs:
                if self.hp.spkr_embedding_path:
                    speaker_embeddings.append(np.load(wav))
                    # If style encoder is used, we still need to load/extract style embedding from audio
                    if self.style_encoder:
                        # Need to load the actual audio for style encoder
                        # The spkr_embedding_path only contains Pyannote embeddings
                        # This is a limitation - we need the original audio file path
                        # For now, raise an error to indicate this limitation
                        raise ValueError("Cannot use pre-computed speaker embeddings with style encoder. "
                                       "Please provide audio files directly (remove --spkr_embedding_path)")
                else:
                    # If wav is a path string
                    if isinstance(wav, str):
                        audio, sr = sf.read(wav)
                        wav = audio # Assumes correct SR or handling elsewhere (infer.py does it)
                    
                    wav = normalize(wav) * 0.95
                    wav_torch = torch.FloatTensor(wav).unsqueeze(0)
                    
                    # Pyannote
                    spk_emb = self.spkr_embedding({'waveform': wav_torch, 'sample_rate': self.hp.sample_rate})
                    speaker_embeddings.append(spk_emb)
                    
                    # Style Encoder (always extract if model was trained with it)
                    if self.style_encoder:
                        style_emb = self.style_encoder(wav_torch.cuda())
                        style_embeddings.append(style_emb)

            speaker_embeddings = torch.tensor(np.array(speaker_embeddings), dtype=torch.float32, device='cuda')
            norm_spkr = F.normalize(speaker_embeddings, dim=-1)
            
            if self.style_encoder and len(style_embeddings) > 0:
                 style_embeddings = torch.cat(style_embeddings, dim=0)
                 norm_style = F.normalize(style_embeddings, dim=-1)
                 combined_spkr = torch.cat([norm_style, norm_spkr], dim=-1)
                 speaker_embedding = self.spkr_linear(combined_spkr)
                 voc_spkr = combined_spkr
                 if self.style_to_vocoder:
                     voc_spkr = self.style_to_vocoder(voc_spkr)
            else:
                 speaker_embedding = self.spkr_linear(norm_spkr)
                 voc_spkr = norm_spkr

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
            synthetic = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, speaker_embedding, phone_masks, prior=prior)
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
            synthetic = self.vocoder(padded_synthetic, voc_spkr)
            outputs = []
            for l, s in zip(lengths, synthetic):
                if self.hp.clean_speech_prior:
                    l = l + self.hp.prior_frame * 256
                outputs.append(s[0, : l].cpu().numpy())
            return outputs
