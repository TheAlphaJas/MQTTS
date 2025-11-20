import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.QuantizeDataset import QuantizeDataset, QuantizeDatasetVal
from data.sampler import RandomBucketSampler
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from modules.vocoder import Vocoder
from modules.style_encoder import StyleEncoder
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
        self.spkr_linear = nn.Linear(512, hp.hidden_size)
        
        # Style Encoder
        self.style_encoder = None
        if hasattr(hp, 'style_encoder_type') and hp.style_encoder_type == 'style_tts2':
            self.style_encoder = StyleEncoder()
            
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
                else:
                     state_dict = ckpt
                
                # Filter keys for style_encoder
                # StyleTTS2 keys usually start with 'style_encoder.'
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('style_encoder.'):
                        new_state_dict[k.replace('style_encoder.', '')] = v
                    elif k.startswith('style_enc.'): # Potential variation
                        new_state_dict[k.replace('style_enc.', '')] = v
                
                if len(new_state_dict) > 0:
                    print(f"Found {len(new_state_dict)} keys for style encoder.")
                    missing, unexpected = self.style_encoder.load_state_dict(new_state_dict, strict=False)
                    print(f"Missing keys: {missing}")
                    print(f"Unexpected keys: {unexpected}")
                else:
                    print("No style encoder keys found in checkpoint. Initializing randomly.")


        if self.hp.pretrained_path:
            self.load()
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

    def load(self):
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=False)

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
        #Deal with speaker embedding
        if self.style_encoder:
             style_embedding = self.style_encoder(batch['audio'])
             speaker_embedding = F.normalize(style_embedding, dim=-1)
        else:
             speaker_embedding = F.normalize(batch['speaker'], dim=-1)
        
        speaker_embedding_proj = self.spkr_linear(F.dropout(speaker_embedding, self.hp.speaker_embed_dropout))
        
        #Deal with phone segments
        phone_features = self.phone_embedding(batch['phone'])
        #Run decoder
        recons_segments = self.TTSdecoder(batch['tts_quantize_input'], phone_features, speaker_embedding_proj,
                                          batch['quantize_mask'], batch['phone_mask'])
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        
        # Verbose logging
        if self.global_step % self.hp.verbose_step == 0 and self.global_step > 0:
             msg = f"Epoch: {self.current_epoch} | Step: {self.global_step} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}"
             print(msg)
             if hasattr(self.hp, 'verbose_file') and self.hp.verbose_file:
                 with open(self.hp.verbose_file, 'a') as f:
                     f.write(f"{msg}\n")

        if hasattr(self.hp, 'fine_tune_vocoder') and self.hp.fine_tune_vocoder:
            # Vocoder Fine-tuning logic
            # Use GT codes: batch['tts_quantize_input'] (or output?)
            # tts_quantize_input has Start Token?
            # Let's use batch['tts_quantize_input'] excluding start token for better alignment?
            # The vocoder typically trained on codes without start token if they represent audio frames.
            # batch['tts_quantize_input'] is padded.
            
            # In validation_step, it uses: self.vocoder(q_s[:, 1:], norm_spkr)
            # q_s is batch['tts_quantize_input'] (from QuantizeDataset).
            # So we should use batch['tts_quantize_input'][:, 1:]
            
            voc_input = batch['tts_quantize_input'][:, 1:]
            
            # Generate audio
            audio_hat = self.vocoder(voc_input, speaker_embedding) # Pass normalized style embedding
            
            # Get GT audio
            audio_gt = batch['audio'].unsqueeze(1) # N, 1, T
            
            # Match lengths
            min_len = min(audio_gt.shape[-1], audio_hat.shape[-1])
            audio_gt = audio_gt[..., :min_len]
            audio_hat = audio_hat[..., :min_len]
            
            # Compute Mel Loss
            # Need self.vocoder.h params
            h = self.vocoder.h
            mel_gt = mel_spectrogram(audio_gt.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
            mel_hat = mel_spectrogram(audio_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
            
            loss_voc = F.l1_loss(mel_hat, mel_gt)
            self.log("train/loss_voc", loss_voc, on_step=True, prog_bar=True)
            
            loss = loss + loss_voc
            
        return loss

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
        spkr, q_s, q_e, phone, ground_truth = batch
        
        # We need audio for style encoder in validation too, but validation_step signature in dataset returns it as ground_truth (5th element)
        # QuantizeDatasetVal returns: (spkr, q_s, q_e, ph, audio)
        # But wait, my modification to QuantizeDatasetVal.getitem returned:
        # (speaker_embedding, quantization_s, quantization_e, phonemes, audio)
        # So batch unpacking:
        # spkr, q_s, q_e, phone, ground_truth = batch
        # This matches.
        
        if self.style_encoder:
             # ground_truth is the audio (N, T)
             # We need to pass it to style_encoder.
             # But ground_truth in validation might be just waveform.
             # style_encoder expects (B, T)
             style_embedding = self.style_encoder(ground_truth)
             norm_spkr = F.normalize(style_embedding, dim=-1)
        else:
             norm_spkr = F.normalize(spkr, dim=-1)
             
        spkr_proj = self.spkr_linear(norm_spkr)
        phone_features = self.phone_embedding(phone)
        recons_segments = self.TTSdecoder(q_s, phone_features, spkr_proj, None, None)
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
            synthetic, infer_attn = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, spkr_proj, phone_mask, prior=None, output_alignment=True)
            synthetic = synthetic[0].unsqueeze(0)
            synthetic = self.vocoder(synthetic, norm_spkr).float()
            #Reconstructed Audio with vocoder
            reconstructed_gt = self.vocoder(q_s[:, 1:], norm_spkr).float()
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
