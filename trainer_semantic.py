import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.QuantizeDataset_semantic import QuantizeDataset, QuantizeDatasetVal
from data.sampler import RandomBucketSampler
from modules.wildttstransformer_semantic import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from modules.vocoder import Vocoder
from modules.style_encoder import StyleEncoder
from modules.text_encoder import SemanticEncoder
from quantizer.meldataset import mel_spectrogram
from torch.utils import data
import pytorch_lightning.core.module as pl
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
plt.switch_backend('agg')

from modules.utmos_predictor import SpeakerEmbeddingSimilarityLoss

# -----------------------------
# Multi-Resolution STFT Loss
# -----------------------------
class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss (spectral convergence + log mag / L1).
    """
    def __init__(self, fft_size, hop_size, win_size, eps=1e-8):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.eps = eps

    def forward(self, x, y):
        # x, y: (B, T)
        # torch.stft expects shape (..., T)
        x_stft = torch.stft(x, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size,
                            return_complex=True, window=torch.hann_window(self.win_size).to(x.device))
        y_stft = torch.stft(y, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size,
                            return_complex=True, window=torch.hann_window(self.win_size).to(y.device))

        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral convergence
        sc_num = torch.norm(y_mag - x_mag, p='fro')
        sc_den = torch.norm(y_mag, p='fro') + self.eps
        sc = sc_num / sc_den

        # L1 magnitude loss
        mag = F.l1_loss(x_mag, y_mag)

        return sc, mag

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss aggregating several STFT settings.
    """
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            resolutions = [
                (1024, 256, 1024),
                (2048, 512, 2048),
                (512, 128, 512),
            ]
        self.losses = nn.ModuleList([STFTLoss(f, h, w) for (f, h, w) in resolutions])

    def forward(self, x, y):
        sc_sum = 0.0
        mag_sum = 0.0
        for loss in self.losses:
            sc, mag = loss(x, y)
            sc_sum += sc
            mag_sum += mag
        sc_mean = sc_sum / len(self.losses)
        mag_mean = mag_sum / len(self.losses)
        return sc_mean, mag_mean

class Wav2TTS(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.data = QuantizeDataset(hp, hp.metapath)
        self.val_data = QuantizeDatasetVal(hp, hp.val_metapath)
        self.TTSdecoder = TTSDecoder(hp, len(self.data.phoneset))
        self.n_decode_codes = self.TTSdecoder.transducer.n_decoder_codes
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.hp.label_smoothing)
        self.phone_embedding = nn.Embedding(len(self.data.phoneset), hp.hidden_size, padding_idx=self.data.phoneset.index('<pad>'))
        
        # Initialize spkr_linear with correct input dimension
        spkr_embed_dim = 512
        
        if hasattr(hp, 'speaker_embedding_dir') and hp.speaker_embedding_dir:
            print(f"[NOTE] Using pre-computed speaker embeddings from {hp.speaker_embedding_dir}")
        else:
            print("[NOTE] Computing speaker embeddings on-the-fly (if Pyannote is available/enabled)")
        
        # Semantic Encoder (SBERT)
        self.semantic_encoder = SemanticEncoder(device='cuda')
        
        # Style Encoder
        self.style_encoder = None
        if hasattr(hp, 'style_encoder_type') and hp.style_encoder_type == 'style_tts2':
            self.style_encoder = StyleEncoder()
            spkr_embed_dim = 128 + 512

            if self.style_encoder and hasattr(hp, 'style_encoder_ckpt') and hp.style_encoder_ckpt:
                print(f"Loading style encoder weights from {hp.style_encoder_ckpt}")
                ckpt = torch.load(hp.style_encoder_ckpt, map_location='cpu')
                if 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'net' in ckpt:
                    state_dict = ckpt['net']
                else:
                    state_dict = ckpt
                
                new_state_dict = {}
                if 'style_encoder' in state_dict and isinstance(state_dict['style_encoder'], dict):
                    print("Found nested 'style_encoder' dict.")
                    temp_dict = state_dict['style_encoder']
                    for k, v in temp_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k.replace('module.', '')] = v
                        else:
                            new_state_dict[k] = v
                else:
                    for k, v in state_dict.items():
                        if k.startswith('style_encoder.'):
                            key_suffix = k.replace('style_encoder.', '')
                            if key_suffix.startswith('module.'):
                                key_suffix = key_suffix.replace('module.', '')
                            new_state_dict[key_suffix] = v
                        elif k.startswith('style_enc.'): 
                            new_state_dict[k.replace('style_enc.', '')] = v
                
                if len(new_state_dict) > 0:
                    print(f"Found {len(new_state_dict)} keys for style encoder.")
                    missing, unexpected = self.style_encoder.load_state_dict(new_state_dict, strict=False)
                    print(f"[NOTE] StyleTTS2 weights loaded from {hp.style_encoder_ckpt}.")
                    if len(missing) > 0:
                        print(f"   Missing keys: {missing}")
                    if len(unexpected) > 0:
                        print(f"   Unexpected keys: {unexpected}")
                else:
                    print("[NOTE] No style encoder keys found in checkpoint. Initializing randomly.")

        self.spkr_linear = nn.Linear(spkr_embed_dim, hp.hidden_size)
        
        # Adapter for Vocoder if using StyleEncoder
        self.style_to_vocoder = None
        if self.style_encoder:
            self.style_to_vocoder = nn.Linear(128 + 512, 512)
            
            if hasattr(hp, 'vocoder_ckpt_path') and hp.vocoder_ckpt_path:
                print(f"Loading style_to_vocoder weights from {hp.vocoder_ckpt_path}")
                try:
                    ckpt = torch.load(hp.vocoder_ckpt_path, map_location='cpu')
                    if 'style_to_vocoder' in ckpt:
                        self.style_to_vocoder.load_state_dict(ckpt['style_to_vocoder'])
                        print("Loaded style_to_vocoder weights.")
                    else:
                        print("[WARNING] 'style_to_vocoder' not found in vocoder checkpoint. Using random initialization.")
                except Exception as e:
                    print(f"[WARNING] Failed to load style_to_vocoder from vocoder ckpt: {e}")

        if self.hp.pretrained_path:
            self.load()
            ckpt_keys = torch.load(self.hp.pretrained_path, map_location='cpu')['state_dict'].keys()
            has_style_enc_in_ckpt = any('style_encoder' in k for k in ckpt_keys)
            
            if has_style_enc_in_ckpt:
                print(f"[NOTE] The provided pretrained_path '{self.hp.pretrained_path}' contained Style Encoder weights.")
                print("       These weights have OVERWRITTEN the StyleTTS2 initialization.")
            else:
                print(f"[NOTE] The provided pretrained_path '{self.hp.pretrained_path}' did NOT contain Style Encoder weights.")
                print("       Keeping the StyleTTS2 initialization (or random if not provided).")
        else:
            self.apply(self.init_weights)
        
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        
        # Vocoder fine-tune / freeze logic (keep evaluation mode for frozen)
        if hasattr(hp, 'fine_tune_vocoder') and hp.fine_tune_vocoder:
            print("Fine-tuning vocoder enabled - gradients will flow through vocoder.")
            self.vocoder.train()
        else:
            print("Vocoder frozen (default).")
            self.vocoder.eval()
            # Freeze vocoder parameters but keep gradient flow flag off (parameters not trainable)
            for param in self.vocoder.parameters():
                param.requires_grad = False
            
            if self.style_to_vocoder:
                self.style_to_vocoder.eval()
                for param in self.style_to_vocoder.parameters():
                    param.requires_grad = False
        
        # ===== INITIALIZE LOSS FUNCTIONS =====
        # Speaker Similarity Loss
        self.use_speaker_similarity_loss = hasattr(hp, 'speaker_similarity_weight') and hp.speaker_similarity_weight > 0
        if self.use_speaker_similarity_loss:
            self.speaker_similarity_loss = SpeakerEmbeddingSimilarityLoss()
            print(f"[INFO] Speaker Similarity Loss enabled with weight: {hp.speaker_similarity_weight}")
        
        # Acoustic losses
        self.stft_loss_fn = MultiResolutionSTFTLoss()
        self.mel_l1_loss = nn.L1Loss()
        self.mel_l2_loss = nn.MSELoss()
        
        # SI-SDR function (negative SI-SDR as loss)
        def si_sdr_fn(x, y, eps=1e-8):
            # x: reference (B, T), y: estimated (B, T)
            # zero-mean
            x_zm = x - x.mean(dim=-1, keepdim=True)
            y_zm = y - y.mean(dim=-1, keepdim=True)
            # scale
            a = (y_zm * x_zm).sum(-1, keepdim=True) / (x_zm.pow(2).sum(-1, keepdim=True) + eps)
            proj = a * x_zm
            noise = y_zm - proj
            sdr = 10.0 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))
            # We return negative SI-SDR (to minimize)
            return -sdr.mean()
        self.si_sdr_fn = si_sdr_fn
        
        # Loss weights (sensible defaults for wild-data TTS)
        # You can override these via hp.*_weight
        self.loss_weights = {
            'ce': getattr(hp, 'ce_weight', 1.0),
            'spk_sim': getattr(hp, 'speaker_similarity_weight', 0.7),
            'mel_l1': getattr(hp, 'mel_l1_weight', 0.0),
            'mel_l2': getattr(hp, 'mel_l2_weight', 0.0),
            'stft': getattr(hp, 'stft_weight', 0.0),
            'sisdr': getattr(hp, 'sisdr_weight', 0.0),
        }
        
        # Gumbel-Softmax temperature for differentiable sampling
        self.gumbel_temperature = getattr(hp, 'gumbel_temperature', 1.0)
        self.use_gumbel = getattr(hp, 'use_gumbel_softmax', True)

        print(f"[INFO] Loss weights - CE: {self.loss_weights['ce']}, Spk_sim: {self.loss_weights['spk_sim']}, MelL1: {self.loss_weights['mel_l1']}, MelL2: {self.loss_weights['mel_l2']}, STFT: {self.loss_weights['stft']}, SI-SDR: {self.loss_weights['sisdr']}")
        print(f"[INFO] Gumbel-Softmax enabled: {self.use_gumbel}, Temperature: {self.gumbel_temperature}")


    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to handle checkpoint incompatibilities.
        """
        model_keys = set(self.state_dict().keys())
        filtered_state_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k.startswith('speaker_similarity_loss.') or \
               k.startswith('semantic_loss_fn.'):
                # keep speaker similarity skip behavior, but no utmos
                skipped_keys.append(k)
                continue
            
            if k in model_keys:
                current_shape = self.state_dict()[k].shape
                if v.shape == current_shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"[WARNING] Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {current_shape}")
                    skipped_keys.append(k)
            else:
                skipped_keys.append(k)
        
        if len(skipped_keys) > 0:
            print(f"[INFO] Skipped {len(skipped_keys)} keys from checkpoint (old/incompatible components)")
            prefixes = {}
            for k in skipped_keys:
                prefix = k.split('.')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            print(f"       Summary: {prefixes}")
        
        missing, unexpected = super().load_state_dict(filtered_state_dict, strict=False)
        
        if len(missing) > 0:
            important_missing = [k for k in missing if not (
                k.startswith('semantic_loss_fn.')
            )]
            
            if len(important_missing) > 0:
                print(f"[WARNING] Missing {len(important_missing)} important keys in checkpoint:")
                print(f"          {important_missing[:5]}...")
        
        if len(unexpected) > 0:
            print(f"[WARNING] {len(unexpected)} unexpected keys")
        
        return missing, unexpected

    def load(self):
        """Load checkpoint using custom load_state_dict."""
        checkpoint = torch.load(self.hp.pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint from {self.hp.pretrained_path}")

    def on_load_checkpoint(self, checkpoint):
        """
        Override Lightning's checkpoint loading to handle optimizer state mismatch.
        """
        state_dict = checkpoint['state_dict']
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not (k.startswith('speaker_similarity_loss.') or 
                    k.startswith('semantic_loss_fn.')):
                if k in self.state_dict():
                    if v.shape == self.state_dict()[k].shape:
                        filtered_state_dict[k] = v
        
        checkpoint['state_dict'] = filtered_state_dict
        
        if 'optimizer_states' in checkpoint:
            print("[INFO] Clearing optimizer states due to model architecture changes")
            checkpoint['optimizer_states'] = []
        
        if 'lr_schedulers' in checkpoint:
            print("[INFO] Clearing LR scheduler states due to model architecture changes")
            checkpoint['lr_schedulers'] = []

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_dataloader(self):
        length = self.data.lengths
        if self.hp.distributed and self.trainer is not None:
            world_size = getattr(self.trainer, 'world_size', getattr(self.trainer, 'num_devices', 1) * getattr(self.trainer, 'num_nodes', 1))
            rank = getattr(self.trainer, 'local_rank', getattr(self.trainer, 'global_rank', 0))
        else:
            world_size = 1
            rank = 0
        sampler = RandomBucketSampler(self.hp.train_bucket_size, length, self.hp.batch_size, drop_last=True, distributed=self.hp.distributed,
                                      world_size=world_size, rank=rank)
        dataset = data.DataLoader(self.data,
                                  num_workers=self.hp.nworkers,
                                  batch_sampler=sampler,
                                  collate_fn=self.data.seqCollate)
        return dataset

    def val_dataloader(self):
        dataset = data.DataLoader(self.val_data,
                                  num_workers=self.hp.nworkers,
                                  shuffle=False)
        return dataset

    def configure_optimizers(self):
        # Only include trainable parameters in optimizer
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"[INFO] Configuring optimizer with {len(trainable_params)} parameter groups")
        print(f"       Total trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer_adam = optim.Adam(trainable_params, lr=self.hp.lr, betas=(self.hp.adam_beta1, self.hp.adam_beta2))
        
        num_training_steps = self.hp.training_step
        num_warmup_steps = self.hp.warmup_step
        num_flat_steps = int(self.hp.optim_flat_percent * num_training_steps)
        
        def lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
            )
        
        scheduler_adam = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer_adam, lambda_lr),
            'interval': 'step'
        }
        return [optimizer_adam], [scheduler_adam]

    def gumbel_softmax_sample(self, logits, temperature=1.0, hard=False):
        """
        Differentiable sampling using Gumbel-Softmax trick.
        
        Args:
            logits: (B, T, n_cluster_groups, n_decoder_codes)
            temperature: Gumbel temperature
            hard: If True, returns one-hot (straight-through estimator)
        
        Returns:
            samples: Soft or hard samples with same shape as logits
        """
        # Add Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        
        # Softmax
        y_soft = F.softmax(gumbels, dim=-1)
        
        if hard:
            # Straight through: forward hard, backward soft
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        return ret

    def training_step(self, batch, batch_idx):
        # Ensure frozen components stay in eval mode
        self.semantic_encoder.eval()

        if self.use_speaker_similarity_loss:
            self.speaker_similarity_loss.eval()
        
        # ===== PREPARE EMBEDDINGS =====
        if self.style_encoder:
            style_embedding = self.style_encoder(batch['audio'])
            style_embedding = F.normalize(style_embedding, dim=-1)
            pyannote_embedding = F.normalize(batch['speaker'], dim=-1)
            speaker_embedding = torch.cat([style_embedding, pyannote_embedding], dim=-1)
        else:
            speaker_embedding = F.normalize(batch['speaker'], dim=-1)
        
        speaker_embedding_proj = self.spkr_linear(F.dropout(speaker_embedding, self.hp.speaker_embed_dropout))
        phone_features = self.phone_embedding(batch['phone'])
        
        # Extract Semantic Embeddings (SBERT)
        raw_texts = batch['text']
        semantic_embedding = self.semantic_encoder(raw_texts)
        
        # ===== FORWARD PASS =====
        recons_segments = self.TTSdecoder(
            batch['tts_quantize_input'], 
            phone_features, 
            speaker_embedding_proj,
            semantic_embedding,
            batch['quantize_mask'], 
            batch['phone_mask']
        )
        
        # ===== COMPUTE CROSS-ENTROPY LOSS =====
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        ce_loss = self.cross_entropy(target, labels)
        
        acc = (target.argmax(-1) == labels).float().mean()
        
        # ===== GENERATE AUDIO WITH GRADIENT FLOW =====
        logits = recons_segments['logits']  # (B, T, n_cluster_groups, n_decoder_codes)
        
        # Use gumbel when enabled and when speaker similarity is being used (so gradients flow to audio for spk loss)
        if self.use_gumbel and self.use_speaker_similarity_loss:
            # Use Gumbel-Softmax for differentiable sampling
            soft_codes = self.gumbel_softmax_sample(
                logits, 
                temperature=self.gumbel_temperature, 
                hard=True  # Straight-through estimator
            )
            
            # Convert soft codes to indices for vocoder
            # soft_codes is (B, T, n_cluster_groups, n_decoder_codes) with one-hot
            voc_input = soft_codes.argmax(dim=-1)  # (B, T, n_cluster_groups)
            
            # Clamp to valid range
            voc_input = torch.clamp(voc_input, 0, self.hp.n_codes - 1)
        else:
            # print("NON DIFF ERROR!!")
            # Standard argmax (non-differentiable)
            with torch.no_grad():
                voc_input = logits.argmax(dim=-1)
                voc_input = torch.clamp(voc_input, 0, self.hp.n_codes - 1)
        
        # Prepare speaker embedding for vocoder
        voc_spkr = speaker_embedding
        if self.style_to_vocoder is not None:
            voc_spkr = self.style_to_vocoder(voc_spkr)
        
        # Generate audio (WITH gradient flow if using Gumbel for differentiability)
        if self.use_gumbel and self.use_speaker_similarity_loss:
            audio_hat = self.vocoder(voc_input, voc_spkr)
        else:
            # print("GUMBEL NOT USED BRUH")
            with torch.no_grad():
                audio_hat = self.vocoder(voc_input, voc_spkr)
        
        # Ground truth audio
        audio_gt = batch['audio']
        if audio_gt.dim() == 2:
            audio_gt = audio_gt.unsqueeze(1)
        
        # Align lengths
        min_len = min(audio_gt.shape[-1], audio_hat.shape[-1])
        audio_gt = audio_gt[..., :min_len]
        audio_hat = audio_hat[..., :min_len]
        
        # ===== COMPUTE PERCEPTUAL / ACOUSTIC LOSSES =====
        losses_dict = {'ce': ce_loss}
        
        # Use autocast for efficiency, but compute mel & si-sdr in float when necessary
            # Speaker similarity loss
        if self.use_speaker_similarity_loss:
            try:
                spk_sim_loss, spk_sim_score = self.speaker_similarity_loss(
                        audio_hat.squeeze(1).float(), 
                        audio_gt.squeeze(1).float(), 
                        sample_rate=self.hp.sample_rate
                )
                losses_dict['spk_sim'] = spk_sim_loss
                    # log speaker similarity score
                self.log("train/speaker_similarity_score", spk_sim_score, on_step=True, prog_bar=True)
            except Exception as e:
                print(f"[WARNING] Speaker similarity loss failed: {e}")
            
            # --- Mel spectrogram reconstruction losses (compute in float) ---
        try:
            mel_args = {
                    'n_fft': getattr(self.hp, 'n_fft', 1024),
                    'num_mels': getattr(self.hp, 'num_mels', 80),
                    'sampling_rate': getattr(self.hp, 'sampling_rate', getattr(self.hp, 'sample_rate', 16000)),
                    'hop_size': getattr(self.hp, 'hop_size', getattr(self.hp, 'hop_size', 256)),
                    'win_size': getattr(self.hp, 'win_size', getattr(self.hp, 'win_size', 1024)),
                    'fmin': getattr(self.hp, 'fmin', 0.0),
                    'fmax': getattr(self.hp, 'fmax', None)
                }
                # mel_spectrogram likely expects (wav, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
                # We'll be defensive about fmax None handling inside mel_spectrogram implementation
            mel_gt = mel_spectrogram(
                    audio_gt.squeeze(1).float(),
                    mel_args['n_fft'],
                    mel_args['num_mels'],
                    mel_args['sampling_rate'],
                    mel_args['hop_size'],
                    mel_args['win_size'],
                    mel_args['fmin'],
                    mel_args['fmax']
            )
            mel_hat = mel_spectrogram(
                    audio_hat.squeeze(1).float(),
                    mel_args['n_fft'],
                    mel_args['num_mels'],
                    mel_args['sampling_rate'],
                    mel_args['hop_size'],
                    mel_args['win_size'],
                    mel_args['fmin'],
                    mel_args['fmax']
            )
                # Align mel lengths (if any)
            if mel_hat.size(-1) != mel_gt.size(-1):
                min_mel = min(mel_hat.size(-1), mel_gt.size(-1))
                mel_hat = mel_hat[..., :min_mel]
                mel_gt = mel_gt[..., :min_mel]
                
            mel_l1 = self.mel_l1_loss(mel_hat, mel_gt)
            mel_l2 = self.mel_l2_loss(mel_hat, mel_gt)
            losses_dict['mel_l1'] = mel_l1
            losses_dict['mel_l2'] = mel_l2
        except Exception as e:
            print(f"[WARNING] Mel spectrogram losses failed: {e}")
            
            # --- STFT multi-resolution loss (operates on waveform tensors) ---
        try:
                # stft loss returns (sc_mean, mag_mean)
            sc_mean, mag_mean = self.stft_loss_fn(audio_hat.squeeze(1).float(), audio_gt.squeeze(1).float())
            stft_loss = sc_mean + mag_mean
            losses_dict['stft'] = stft_loss
        except Exception as e:
            print(f"[WARNING] STFT loss failed: {e}")
            
            # --- SI-SDR loss (waveform-level) ---
        try:
            sisdr_loss = self.si_sdr_fn(audio_gt.squeeze(1).float(), audio_hat.squeeze(1).float())
            losses_dict['sisdr'] = sisdr_loss
        except Exception as e:
            print(f"[WARNING] SI-SDR loss failed: {e}")
            
            # Combine losses with gradient normalization (existing logic)
        active_losses = {k: v for k, v in losses_dict.items() if self.loss_weights.get(k, 0) > 0}
            
        if len(active_losses) == 0:
            total_loss = ce_loss
        else:
            normalized_losses = []
            loss_magnitudes = []
                
            for loss_name, loss_value in active_losses.items():
                    # Ensure scalar tensor
                if isinstance(loss_value, torch.Tensor):
                        magnitude = float(loss_value.detach().cpu().item())
                else:
                    try:
                        magnitude = float(loss_value)
                    except:
                        magnitude = 1.0
                weight = self.loss_weights.get(loss_name, 1.0)
                    
                if magnitude > 1e-8:
                    norm_loss = (loss_value / magnitude) * weight
                    normalized_losses.append(norm_loss)
                    loss_magnitudes.append(magnitude)
                else:
                    normalized_losses.append(loss_value * weight)
                    loss_magnitudes.append(1.0)
                
            mean_magnitude = sum(loss_magnitudes) / len(loss_magnitudes)
            total_loss = mean_magnitude * sum(normalized_losses)
                
                # Log individual losses
            for loss_name, loss_value in active_losses.items():
                try:
                    self.log(f"train/loss_{loss_name}", loss_value, on_step=True, prog_bar=False)
                    self.log(f"train/loss_{loss_name}_magnitude", float(loss_value.detach().cpu().item()) if isinstance(loss_value, torch.Tensor) else float(loss_value), on_step=True, prog_bar=False)
                except Exception:
                        # Fallback logging
                    try:
                        self.log(f"train/loss_{loss_name}", float(loss_value), on_step=True, prog_bar=False)
                    except:
                        pass
            
            # Always log the combined loss and accuracy
        self.log("train/loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        
        return total_loss

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            try:
                total_samples = len(self.data)
                batch_size = self.hp.batch_size
                world_size = getattr(self.trainer, 'world_size', 1)
                steps = total_samples // (batch_size * world_size)
                print(f"\n[INFO] Epoch {self.current_epoch} started. Total samples: {total_samples}. Estimated steps per epoch: {steps}")
            except:
                pass

    def on_validation_epoch_start(self):
        start_point, half = 4, self.hp.sample_num // 2
        if self.hp.sample_num > 0:
            self.sample_idxs = list(range(start_point, start_point + half)) + \
                np.random.randint(low=start_point + half, high=len(self.val_data), size=self.hp.sample_num//2).tolist()
        else:
            self.sample_idxs = []

    def validation_step(self, batch, batch_idx):
        spkr, q_s, q_e, phone, ground_truth, raw_text = batch
        
        if isinstance(raw_text, tuple):
            raw_text = list(raw_text)
        semantic_embedding = self.semantic_encoder(raw_text)
        
        if self.style_encoder:
            style_embedding = self.style_encoder(ground_truth)
            style_norm = F.normalize(style_embedding, dim=-1)
            pyannote_norm = F.normalize(spkr, dim=-1)
            norm_spkr = torch.cat([style_norm, pyannote_norm], dim=-1)
        else:
            norm_spkr = F.normalize(spkr, dim=-1)
             
        spkr_proj = self.spkr_linear(norm_spkr)
        phone_features = self.phone_embedding(phone)
        recons_segments = self.TTSdecoder(q_s, phone_features, spkr_proj, semantic_embedding, None, None)
        target = recons_segments['logits'].view(-1, self.n_decode_codes)
        labels = q_e.view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("val/loss", loss, on_epoch=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, logger=True)

        if batch_idx in self.sample_idxs:
            batch_idx = self.sample_idxs.index(batch_idx)
            phone_mask = torch.full((phone_features.size(0), phone_features.size(1)), False, dtype=torch.bool, device=phone_features.device)
            synthetic, infer_attn = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, spkr_proj, semantic_embedding, phone_mask, prior=None, output_alignment=True)
            synthetic = synthetic[0].unsqueeze(0)
            
            voc_input_val = q_s[:, 1:].clone()
            voc_input_val[voc_input_val >= self.hp.n_codes] = 0

            voc_spkr = norm_spkr
            if self.style_to_vocoder is not None:
                voc_spkr = self.style_to_vocoder(voc_spkr)
            
            synthetic = self.vocoder(synthetic, voc_spkr).float()
            reconstructed_gt = self.vocoder(voc_input_val, voc_spkr).float()
            
            sw = self.logger.experiment
            sw.add_audio(f'generated/{batch_idx}', synthetic, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'vocoder-reconstructed/{batch_idx}', reconstructed_gt, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'groundtruth/{batch_idx}', ground_truth[0], self.global_step, self.hp.sample_rate)

            self.plot_attn(recons_segments['encoder_attention'], f'enc-attn/{batch_idx}', (10, 10))
            self.plot_attn(recons_segments['decoder_attention'], f'dec-attn/{batch_idx}', (10, 10))
            self.plot_attn([recons_segments['alignment']], f'train-alignment/{batch_idx}', (10, 10))
            self.plot_attn([infer_attn.unsqueeze(0)], f'infer-alignment/{batch_idx}', (10, 10))

    def plot_attn(self, attns, prefix, figsize):
        nheads = attns[0].size(1)
        fig, axs = plt.subplots(len(attns), nheads, constrained_layout=True, figsize=figsize)
        if len(attns) == 1 and nheads == 1:
            axs = [[axs]]
        elif len(attns) == 1 or nheads == 1:
            axs = [axs]
        for i, attn in enumerate(attns):
            attn = attn.float().cpu().numpy()
            for j, head_attn in enumerate(attn[0]):
                axs[i][j].matshow(head_attn, aspect="auto", origin="lower", interpolation='none')
                if i != 0 or j != 0:
                    axs[i][j].axis('off')
        self.logger.experiment.add_figure(prefix, fig, self.global_step)
        plt.close()