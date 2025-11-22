import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")
import sys


import itertools
import os
# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset_joint import MelDatasetJoint, mel_spectrogram, get_dataset_filelist
import warnings
warnings.filterwarnings("ignore")
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, Encoder, Quantizer
from modules.style_encoder import StyleEncoder

try:
    from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
except:
    from .utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def load_style_encoder_weights(model, ckpt_path):
    print(f"Loading style encoder weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
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
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Style Encoder loaded. Missing: {missing}, Unexpected: {unexpected}")
    else:
        print("No style encoder keys found in checkpoint.")

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)
    
    # Style Encoder Setup
    style_encoder = StyleEncoder().to(device)
    # Adapter: StyleTTS2 (128) + Pyannote (512) -> Vocoder (512)
    # We will concatenate them: 128 + 512 = 640 -> 512
    style_to_vocoder = nn.Linear(128 + 512, 512).to(device)
    
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # Load Checkpoint
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    
    steps = 0
    last_epoch = -1
    
    if cp_g is None or cp_do is None:
        state_dict_do = None
        if a.checkpoint_file: # Load base pretrained (e.g. original MQTTS)
            print(f"Loading pretrained base checkpoint from {a.checkpoint_file}")
            state_dict_g = load_checkpoint(a.checkpoint_file, device)
            encoder.load_state_dict(state_dict_g['encoder'])
            quantizer.load_state_dict(state_dict_g['quantizer'])
            generator.load_state_dict(state_dict_g['generator'])
            
            # Load style encoder pretrained if provided and not in checkpoint
            if a.style_encoder_ckpt:
                load_style_encoder_weights(style_encoder, a.style_encoder_ckpt)
    else:
        print(f"Resuming from {cp_g}")
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        
        generator.load_state_dict(state_dict_g['generator'])
        encoder.load_state_dict(state_dict_g['encoder'])
        quantizer.load_state_dict(state_dict_g['quantizer'])
        
        # Handle dimension mismatch for generator.spkr_linear if loading old checkpoint
        # Old Generator expects 512 input. Our new logic provides 512 input (projected from 640).
        # Wait, the Generator itself has a spkr_linear layer: nn.Linear(512, 512).
        # If we project 640->512 using style_to_vocoder, the Generator input is 512.
        # So the Generator structure is UNCHANGED. We can safely load it.
        # The only new component is style_to_vocoder (which is external to Generator).
        
        generator.load_state_dict(state_dict_g['generator'])
        
        if 'style_encoder' in state_dict_g:
            style_encoder.load_state_dict(state_dict_g['style_encoder'])
            # Check if style_to_vocoder dimensions match
            # If previous training was just style encoder (128->512), this will fail if we now do (640->512)
            # We should wrap in try/except or check shape
            try:
                style_to_vocoder.load_state_dict(state_dict_g['style_to_vocoder'])
            except RuntimeError as e:
                print(f"Could not load style_to_vocoder weights: {e}. Initializing randomly.")
        else:
            # If resuming but style encoder wasn't saved (migrating), try loading from init arg
            if a.style_encoder_ckpt:
                load_style_encoder_weights(style_encoder, a.style_encoder_ckpt)
        
        if cp_g: # Only load optimizers if we loaded a full checkpoint (not just base)
             if 'mpd' in state_dict_do:
                 mpd.load_state_dict(state_dict_do['mpd'])
             if 'msd' in state_dict_do:
                 msd.load_state_dict(state_dict_do['msd'])
        
             steps = state_dict_do['steps'] + 1
             last_epoch = state_dict_do['epoch']
    
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(quantizer, device_ids=[rank]).to(device)
        style_encoder = DistributedDataParallel(style_encoder, device_ids=[rank]).to(device)
        style_to_vocoder = DistributedDataParallel(style_to_vocoder, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(itertools.chain(generator.parameters(), encoder.parameters(), 
                                               quantizer.parameters(), style_encoder.parameters(),
                                               style_to_vocoder.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        try:
            optim_g.load_state_dict(state_dict_do['optim_g'])
            optim_d.load_state_dict(state_dict_do['optim_d'])
        except:
            print("Warning: Optimizer state mismatch (likely due to new parameters). Resetting optimizers.")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDatasetJoint(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, style_segment_size=32000,
                          speaker_embedding_dir=a.speaker_embedding_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        validset = MelDatasetJoint(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir, style_segment_size=32000,
                              speaker_embedding_dir=a.speaker_embedding_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    encoder.train()
    quantizer.train()
    style_encoder.train()
    style_to_vocoder.train()
    mpd.train()
    msd.train()
    
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            
            # y is audio target, spkr_audio is for style, pyannote_emb is pre-extracted
            x, y, _, y_mel, spkr_audio, pyannote_emb = batch
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            spkr_audio = spkr_audio.to(device, non_blocking=True)
            pyannote_emb = pyannote_emb.to(device, non_blocking=True)
            
            y = y.unsqueeze(1) # B, 1, T

            # Compute Style Embedding
            style_emb = style_encoder(spkr_audio) # B, 128
            # Normalize
            style_emb_norm = F.normalize(style_emb, dim=-1)
            
            # Concatenate with Pyannote
            # pyannote_emb is (B, 512)
            combined_emb = torch.cat([style_emb_norm, pyannote_emb], dim=-1) # B, 640
            
            # Project to Vocoder Dim
            spkr_proj = style_to_vocoder(combined_emb) # B, 512

            c = encoder(y)
            q, loss_q, c = quantizer(c)
            
            y_g_hat = generator(q, spkr_proj)
            
            # Trim audio to min length
            min_audio_len = min(y.size(2), y_g_hat.size(2))
            y = y[:, :, :min_audio_len]
            y_g_hat = y_g_hat[:, :, :min_audio_len]
            
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            # Trim to min length to avoid mismatches (e.g. 40 vs 41 frames)
            min_mel_len = min(y_mel.size(2), y_g_hat_mel.size(2))
            y_mel = y_mel[:, :, :min_mel_len]
            y_g_hat_mel = y_g_hat_mel[:, :, :min_mel_len]
            
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_q * 10

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Loss Q : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, loss_q, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                     'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict(),
                                     'quantizer': (quantizer.module if h.num_gpus > 1 else quantizer).state_dict(),
                                     'style_encoder': (style_encoder.module if h.num_gpus > 1 else style_encoder).state_dict(),
                                     'style_to_vocoder': (style_to_vocoder.module if h.num_gpus > 1 else style_to_vocoder).state_dict()
                                     })
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    encoder.eval()
                    quantizer.eval()
                    style_encoder.eval()
                    style_to_vocoder.eval()
                    
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel, spkr_audio, pyannote_emb = batch
                            
                            spkr_audio = spkr_audio.to(device)
                            pyannote_emb = pyannote_emb.to(device)
                            
                            style_emb = style_encoder(spkr_audio)
                            style_emb_norm = F.normalize(style_emb, dim=-1)
                            
                            combined_emb = torch.cat([style_emb_norm, pyannote_emb], dim=-1)
                            spkr_proj = style_to_vocoder(combined_emb)

                            c = encoder(y.to(device).unsqueeze(1))
                            q, loss_q, c = quantizer(c)
                            
                            y_g_hat = generator(q, spkr_proj)
                            y_mel = y_mel.to(device)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                            val_err_tot += F.l1_loss(y_mel[:, :, :i_size], y_g_hat_mel[:, :, :i_size]).item()

                            if j <= 8:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()
                    encoder.train()
                    quantizer.train()
                    style_encoder.train()
                    style_to_vocoder.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='../../../imp_back/datasets/audios')
    parser.add_argument('--input_mels_dir', default=None)
    parser.add_argument('--input_training_file', default='../../../imp_back/datasets/training.txt')
    parser.add_argument('--input_validation_file', default='../../../imp_back/datasets/validation.txt')
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--training_epochs', default=200, type=int)
    parser.add_argument('--stdout_interval', default=10, type=int)
    parser.add_argument('--checkpoint_interval', default=500, type=int)
    parser.add_argument('--summary_interval', default=10, type=int)
    parser.add_argument('--validation_interval', default=10000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--style_encoder_ckpt', default=None, help="Path to StyleTTS2 checkpoint for Style Encoder initialization")
    parser.add_argument('--speaker_embedding_dir', default=None, help="Path to directory containing pre-computed Pyannote embeddings")

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()

