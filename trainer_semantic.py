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
from modules.utmos_predictor import UTMOSPredictor, SpeakerEmbeddingSimilarityLoss
from quantizer.meldataset import mel_spectrogram
from torch.utils import data
import pytorch_lightning.core.module as pl
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
        self.semantic_encoder = SemanticEncoder(device='cuda') # Will be moved to correct device by Lightning
        
        # Style Encoder
        self.style_encoder = None
        if hasattr(hp, 'style_encoder_type') and hp.style_encoder_type == 'style_tts2':
            self.style_encoder = StyleEncoder()
            spkr_embed_dim = 128 # StyleTTS2 style encoder output dim
            
            # Modified: Integration of BOTH Pyannote (512) and StyleTTS2 (128)
            # If we use style_encoder, we will concatenate its output with Pyannote embedding.
            # So spkr_embed_dim should be 128 + 512 = 640 if we are using the concatenated version for spkr_linear.
            # HOWEVER, self.spkr_linear projects to hidden_size (768).
            # In original MQTTS, spkr_linear input is 512 (Pyannote).
            
            # Decision: spkr_linear will take the COMBINED embedding (640) -> hidden_size.
            spkr_embed_dim = 128 + 512

            if self.style_encoder and hasattr(hp, 'style_encoder_ckpt') and hp.style_encoder_ckpt:
                print(f"Loading style encoder weights from {hp.style_encoder_ckpt}")
                ckpt = torch.load(hp.style_encoder_ckpt, map_location='cpu')
                # Handle state dict keys
                # Check if ckpt is full model or just encoder
                if 'model_state_dict' in ckpt:
                     # StyleTTS2 generic checkpoint
                     state_dict = ckpt['model_state_dict']
                elif 'model' in ckpt:
                     # Sometimes saved as model
                     state_dict = ckpt['model']
                elif 'net' in ckpt:
                     state_dict = ckpt['net']
                else:
                     state_dict = ckpt
                
                # Filter keys for style_encoder
                # StyleTTS2 keys usually start with 'style_encoder.'
                new_state_dict = {}
                
                # Check if state_dict has nested style_encoder or if keys start with style_encoder.
                if 'style_encoder' in state_dict and isinstance(state_dict['style_encoder'], dict):
                     print("Found nested 'style_encoder' dict.")
                     temp_dict = state_dict['style_encoder']
                     for k, v in temp_dict.items():
                         # Strip module. prefix if present
                         if k.startswith('module.'):
                             new_state_dict[k.replace('module.', '')] = v
                         else:
                             new_state_dict[k] = v
                else:
                     # Flat dictionary, look for prefixes
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
        # Vocoder expects 512 dim.
        # We concatenate Style (128) + Pyannote (512) = 640.
        self.style_to_vocoder = None
        if self.style_encoder:
            self.style_to_vocoder = nn.Linear(128 + 512, 512)
            
            # Attempt to load style_to_vocoder weights from vocoder_ckpt_path
            # The vocoder_ckpt_path (from train_joint.py) contains 'style_to_vocoder' state_dict
            if hasattr(hp, 'vocoder_ckpt_path') and hp.vocoder_ckpt_path:
                 print(f"Loading style_to_vocoder weights from {hp.vocoder_ckpt_path}")
                 try:
                     ckpt = torch.load(hp.vocoder_ckpt_path, map_location='cpu')
                     # Check key
                     if 'style_to_vocoder' in ckpt:
                         self.style_to_vocoder.load_state_dict(ckpt['style_to_vocoder'])
                         print("Loaded style_to_vocoder weights.")
                     else:
                         print("[WARNING] 'style_to_vocoder' not found in vocoder checkpoint. Using random initialization.")
                 except Exception as e:
                     print(f"[WARNING] Failed to load style_to_vocoder from vocoder ckpt: {e}")

        if self.hp.pretrained_path:
            self.load()
            # Check if the loaded checkpoint actually had style encoder weights (likely not if it was old MQTTS)
            # If we loaded a checkpoint that DID include style encoder (e.g. resuming a semantic run), it might have overwritten what we loaded above.
            # We should inform the user.
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
        
        # Re-load style encoder if it was overwritten by pretrained_path (unlikely if strict=False and keys don't match, but safe to check)
        # Actually, load() calls load_state_dict(..., strict=False).
        # If pretrained_path is MQTTS, it won't have style_encoder keys, so style_encoder remains as initialized above.
        # If pretrained_path IS a new checkpoint that HAS style_encoder, it will overwrite what we loaded above.
        # This is generally desired behavior (resume training).
        # But user wants: 1. StyleTTS2 ckpt for StyleEncoder, 2. MQTTS ckpt for the rest.
        # So we are good: 
        #   Step 1: Initialize StyleEncoder and load StyleTTS2 weights.
        #   Step 2: Load MQTTS weights (which doesn't have StyleEncoder keys) into the rest of the model.
        #   Result: Mixed model. 
        
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        
        if hasattr(hp, 'fine_tune_vocoder') and hp.fine_tune_vocoder:
            print("Fine-tuning vocoder enabled.")
            self.vocoder.train()
            # We do NOT remove weight norm if we are training, usually.
            # But if the loaded checkpoint had weight norm removed, we might need to add it back or just train without it.
            # Given we load a pre-trained vocoder, it likely has weight norm (unless it was saved after removal).
            # Modules like Conv1d default don't have weight norm unless applied.
            # The Generator class uses weight_norm wrapper.
            # If we load state_dict, we need to match structure.
            # Assuming we just leave it as is for fine-tuning.
        else:
            self.vocoder.eval()
            self.vocoder.generator.remove_weight_norm()
            for param in self.vocoder.parameters():
                param.requires_grad = False
            
            # Also freeze style_to_vocoder if not fine-tuning
            if self.style_to_vocoder:
                self.style_to_vocoder.eval()
                for param in self.style_to_vocoder.parameters():
                    param.requires_grad = False
        
        # Initialize additional loss functions for optimization
        # 1. Speaker Embedding Similarity Loss
        self.use_speaker_similarity_loss = hasattr(hp, 'speaker_similarity_weight') and hp.speaker_similarity_weight > 0
        if self.use_speaker_similarity_loss:
            self.speaker_similarity_loss = SpeakerEmbeddingSimilarityLoss()
            print(f"Speaker Embedding Similarity Loss enabled with weight: {hp.speaker_similarity_weight}")
        
        # 2. UTMOS MOS Score Loss
        self.use_utmos_loss = hasattr(hp, 'utmos_weight') and hp.utmos_weight > 0
        if self.use_utmos_loss:
            utmos_ckpt = getattr(hp, 'utmos_ckpt_path', None)
            if utmos_ckpt and os.path.exists(utmos_ckpt):
                self.utmos_predictor = UTMOSPredictor(utmos_ckpt, device='cuda')
                # Check if model actually loaded (it sets utmos_model to None on failure)
                if hasattr(self.utmos_predictor, 'utmos_model') and self.utmos_predictor.utmos_model is None:
                    print("[WARNING] UTMOS model failed to load (see errors above). UTMOS loss will be disabled.")
                    self.use_utmos_loss = False
                else:
                    print(f"UTMOS Loss enabled with weight: {hp.utmos_weight}")
            else:
                print(f"[WARNING] UTMOS checkpoint not found at {utmos_ckpt}. UTMOS loss will be disabled.")
                self.use_utmos_loss = False

    def load(self):
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        model_dict = self.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    print(f"Skipping loading parameter {k} due to shape mismatch. Checkpoint: {v.shape}, Model: {model_dict[k].shape}")
                    continue
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=False)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # PyTorch 2.1: Manually zero out padding index if it exists
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
        # PyTorch Lightning 2.x: world_size and local_rank are still available
        # Use num_devices * num_nodes for world_size, and global_rank for rank
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
        optimizer_adam = optim.Adam(self.parameters(), lr=self.hp.lr, betas=(self.hp.adam_beta1, self.hp.adam_beta2))
        #Learning rate scheduler
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

    def training_step(self, batch, batch_idx):
        # Ensure semantic encoder is in eval mode (to disable dropout etc even if Lightning set it to train)
        self.semantic_encoder.eval()
        
        #Deal with speaker embedding
        if self.style_encoder:
             # For StyleEncoder, we need audio.
             # batch['audio'] is available.
             style_embedding = self.style_encoder(batch['audio'])
             style_embedding = F.normalize(style_embedding, dim=-1)
             
             # We also need Pyannote embedding.
             # batch['speaker'] contains the pre-computed embedding (Pyannote).
             pyannote_embedding = F.normalize(batch['speaker'], dim=-1)
             
             # Concatenate
             speaker_embedding = torch.cat([style_embedding, pyannote_embedding], dim=-1) # 640
        else:
             speaker_embedding = F.normalize(batch['speaker'], dim=-1)
        
        speaker_embedding_proj = self.spkr_linear(F.dropout(speaker_embedding, self.hp.speaker_embed_dropout))
        
        #Deal with phone segments
        phone_features = self.phone_embedding(batch['phone'])
        
        # Extract Semantic Embeddings (SBERT)
        # Batch has raw text in 'text' key if we updated seqCollate, but wait.
        # Lightning passes `batch` which is the output of seqCollate.
        # In QuantizeDataset.seqCollate, I added 'text' list.
        # It is NOT converted to tensor, so it's a list of strings.
        raw_texts = batch['text']
        semantic_embedding = self.semantic_encoder(raw_texts) # (B, 384)
        
        #Run decoder
        recons_segments = self.TTSdecoder(batch['tts_quantize_input'], phone_features, speaker_embedding_proj,
                                          semantic_embedding,
                                          batch['quantize_mask'], batch['phone_mask'])
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        
        # Store base loss for verbose logging
        base_loss = loss.item()
        
        # Additional losses will be computed after generating audio (if needed)
        additional_losses = {}
        
        # Verbose logging
        if self.global_step % self.hp.verbose_step == 0 and self.global_step > 0:
             msg = f"Epoch: {self.current_epoch} | Step: {self.global_step} | Loss: {base_loss:.4f} | Acc: {acc.item():.4f}"
             print(msg)
             if hasattr(self.hp, 'verbose_file') and self.hp.verbose_file:
                 with open(self.hp.verbose_file, 'a') as f:
                     f.write(f"{msg}\n")

        # Generate audio for additional losses (speaker similarity & UTMOS)
        # We need to generate audio regardless of vocoder fine-tuning if these losses are enabled
        audio_hat = None
        audio_gt = None
        voc_spkr = None
        
        if self.use_speaker_similarity_loss or self.use_utmos_loss or (hasattr(self.hp, 'fine_tune_vocoder') and self.hp.fine_tune_vocoder):
            # Use PREDICTED codes to generate audio (gradients will flow back to TTS model)
            # recons_segments['logits'] has shape: (N, T, n_cluster_groups, n_decoder_codes)
            # where n_cluster_groups = 4, n_decoder_codes = n_codes + special tokens
            
            try:
                # Get predicted codes by taking argmax over last dimension
                # Result shape: (N, T, 4)
                predicted_codes = recons_segments['logits'].argmax(dim=-1)
                
                # Filter out padding using the mask
                # batch['quantize_mask']: (N, T) - True where padded
                # We need to keep the full sequence with padding for vocoder
                # but clamp the codes to valid range
                
                # Clone to avoid in-place modification issues with autograd
                voc_input = predicted_codes.clone()
                
                # Clamp to valid code range [0, n_codes-1]
                # Any predicted special tokens (>=n_codes) get clamped to n_codes-1
                voc_input = torch.clamp(voc_input, 0, self.hp.n_codes - 1)
                
                # Prepare speaker embedding for vocoder
                voc_spkr = speaker_embedding  # This is now 640 dim if style_encoder is active
                
                if self.style_to_vocoder is not None:
                    voc_spkr = self.style_to_vocoder(voc_spkr)  # 640 -> 512
                
                # Generate audio from predicted codes
                # voc_input shape: (N, T, 4) - exactly what vocoder expects
                audio_hat = self.vocoder(voc_input, voc_spkr)
                
                # Get GT audio
                audio_gt = batch['audio']
                if audio_gt.dim() == 2:
                    audio_gt = audio_gt.unsqueeze(1)  # N, 1, T
                
                # Match lengths
                min_len = min(audio_gt.shape[-1], audio_hat.shape[-1])
                audio_gt = audio_gt[..., :min_len]
                audio_hat = audio_hat[..., :min_len]
                
            except Exception as e:
                print(f"[WARNING] Failed to generate audio for additional losses: {e}")
                import traceback
                traceback.print_exc()
                audio_hat = None
                audio_gt = None
        
        # Compute additional losses
        # 1. Speaker Embedding Similarity Loss
        if self.use_speaker_similarity_loss and audio_hat is not None and audio_gt is not None:
            try:
                spk_sim_loss, spk_sim_score = self.speaker_similarity_loss(
                    audio_hat.squeeze(1), 
                    audio_gt.squeeze(1), 
                    sample_rate=self.hp.sample_rate
                )
                weighted_spk_loss = spk_sim_loss * self.hp.speaker_similarity_weight
                loss = loss + weighted_spk_loss
                
                self.log("train/speaker_similarity_loss", spk_sim_loss, on_step=True, prog_bar=False)
                self.log("train/speaker_similarity_score", spk_sim_score, on_step=True, prog_bar=False)
                additional_losses['spk_sim'] = spk_sim_loss.item()
                
            except Exception as e:
                print(f"[WARNING] Speaker similarity loss computation failed: {e}")
        
        # 2. UTMOS MOS Score Loss
        if self.use_utmos_loss and audio_hat is not None:
            try:
                mos_loss, mos_score = self.utmos_predictor.compute_mos_loss(
                    audio_hat.squeeze(1),
                    sample_rate=self.hp.sample_rate,
                    target_mos=getattr(self.hp, 'target_mos', 5.0)
                )
                weighted_mos_loss = mos_loss * self.hp.utmos_weight
                loss = loss + weighted_mos_loss
                
                self.log("train/utmos_loss", mos_loss, on_step=True, prog_bar=False)
                self.log("train/predicted_mos", mos_score, on_step=True, prog_bar=True)
                additional_losses['utmos'] = mos_loss.item()
                
            except Exception as e:
                print(f"[WARNING] UTMOS loss computation failed: {e}")
        
        if hasattr(self.hp, 'fine_tune_vocoder') and self.hp.fine_tune_vocoder:
            # Vocoder Fine-tuning logic using GT codes
            # We already generated audio_hat from GT codes above (if additional losses are enabled)
            # If not, we need to generate it here
            
            if audio_hat is None:
                # Generate audio (wasn't generated for additional losses)
                voc_input = batch['tts_quantize_input'][:, 1:].clone()
                voc_input[voc_input >= self.hp.n_codes] = 0
                
                # Prepare speaker embedding
                voc_spkr = speaker_embedding
                if self.style_to_vocoder is not None:
                    voc_spkr = self.style_to_vocoder(voc_spkr)
                
                audio_hat = self.vocoder(voc_input, voc_spkr)
                
                # Get GT audio
                audio_gt = batch['audio']
                if audio_gt.dim() == 2:
                    audio_gt = audio_gt.unsqueeze(1)
                
                # Match lengths
                min_len = min(audio_gt.shape[-1], audio_hat.shape[-1])
                audio_gt = audio_gt[..., :min_len]
                audio_hat = audio_hat[..., :min_len]
            
            # Compute Mel Loss using the generated audio
            if audio_hat is not None and audio_gt is not None:
                h = self.vocoder.h
                mel_gt = mel_spectrogram(audio_gt.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                mel_hat = mel_spectrogram(audio_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                
                # Trim to min length to avoid mismatches
                min_mel_len = min(mel_gt.size(2), mel_hat.size(2))
                mel_gt = mel_gt[:, :, :min_mel_len]
                mel_hat = mel_hat[:, :, :min_mel_len]
                
                loss_voc = F.l1_loss(mel_hat, mel_gt)
                self.log("train/loss_voc", loss_voc, on_step=True, prog_bar=True)
                
                loss = loss + loss_voc
                additional_losses['voc'] = loss_voc.item()
        
        # Enhanced verbose logging with additional losses
        if self.global_step % self.hp.verbose_step == 0 and self.global_step > 0 and len(additional_losses) > 0:
            extra_msg = " | ".join([f"{k}: {v:.4f}" for k, v in additional_losses.items()])
            msg = f"Additional losses: {extra_msg}"
            print(msg)
            if hasattr(self.hp, 'verbose_file') and self.hp.verbose_file:
                with open(self.hp.verbose_file, 'a') as f:
                    f.write(f"{msg}\n")
            
        return loss

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
             try:
                 # Estimate steps
                 total_samples = len(self.data)
                 batch_size = self.hp.batch_size
                 world_size = getattr(self.trainer, 'world_size', 1)
                 # Approximate steps
                 steps = total_samples // (batch_size * world_size)
                 print(f"\n[INFO] Epoch {self.current_epoch} started. Total samples: {total_samples}. Estimated steps per epoch: {steps}")
             except:
                 pass

    def on_validation_epoch_start(self):
        #For the first half samples, and random choose the rest half
        start_point, half = 4, self.hp.sample_num // 2
        if self.hp.sample_num > 0:
            self.sample_idxs = list(range(start_point, start_point + half)) + \
                np.random.randint(low=start_point + half, high=len(self.val_data), size=self.hp.sample_num//2).tolist()
        else:
            self.sample_idxs = []

    def validation_step(self, batch, batch_idx):
        #Batch size = 1
        spkr, q_s, q_e, phone, ground_truth, raw_text = batch
        
        # Extract Semantic Embedding
        # raw_text is a tuple/list of 1 string (batch size 1)
        if isinstance(raw_text, tuple):
             raw_text = list(raw_text)
        semantic_embedding = self.semantic_encoder(raw_text)
        
        if self.style_encoder:
             # ground_truth is the audio (N, T)
             style_embedding = self.style_encoder(ground_truth)
             style_norm = F.normalize(style_embedding, dim=-1)
             
             # batch['speaker'] (spkr) is Pyannote
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

        #Run inference with bs = 1
        if batch_idx in self.sample_idxs:
            batch_idx = self.sample_idxs.index(batch_idx)
            phone_mask = torch.full((phone_features.size(0), phone_features.size(1)), False, dtype=torch.bool, device=phone_features.device)
            synthetic, infer_attn = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, spkr_proj, semantic_embedding, phone_mask, prior=None, output_alignment=True)
            synthetic = synthetic[0].unsqueeze(0)
            
            # Handle padding for vocoder input in validation as well
            # q_s has padding value n_codes.
            voc_input_val = q_s[:, 1:].clone()
            voc_input_val[voc_input_val >= self.hp.n_codes] = 0

            voc_spkr = norm_spkr
            if self.style_to_vocoder is not None:
                 voc_spkr = self.style_to_vocoder(voc_spkr)
            
            synthetic = self.vocoder(synthetic, voc_spkr).float()
            #Reconstructed Audio with vocoder
            reconstructed_gt = self.vocoder(voc_input_val, voc_spkr).float()
            #Write files
            sw = self.logger.experiment
            sw.add_audio(f'generated/{batch_idx}', synthetic, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'vocoder-reconstructed/{batch_idx}', reconstructed_gt, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'groundtruth/{batch_idx}', ground_truth[0], self.global_step, self.hp.sample_rate)

            #Plot attentions
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
        for i, attn in enumerate(attns): #Each layers
            attn = attn.float().cpu().numpy()
            for j, head_attn in enumerate(attn[0]):
                axs[i][j].matshow(head_attn, aspect="auto", origin="lower", interpolation='none')
                if i != 0 or j != 0:
                    axs[i][j].axis('off')
        self.logger.experiment.add_figure(prefix, fig, self.global_step)
        plt.close()
