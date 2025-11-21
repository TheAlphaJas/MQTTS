import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from pyannote.audio import Inference

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDatasetJoint(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, style_segment_size=32000):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.style_segment_size = style_segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.spkr_embedding = Inference("pyannote/embedding", window="whole")

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            try:
                audio, sampling_rate = load_wav(filename)
                audio = audio / MAX_WAV_VALUE
                if not self.fine_tuning:
                    audio = normalize(audio) * 0.95
            except:
                print (f"Error on audio: {filename}")
                audio = np.random.normal(size=(160000,)) * 0.05
                sampling_rate = self.sampling_rate
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0) # 1, T
        
        # Extract Pyannote embedding from the full audio
        # Note: Inference usually takes raw waveform as torch tensor or numpy
        # MQTTS original uses {'waveform': ..., 'sample_rate': ...}
        try:
            pyannote_emb = self.spkr_embedding({'waveform': audio, 'sample_rate': self.sampling_rate})
            pyannote_emb = torch.from_numpy(pyannote_emb).float()
            pyannote_emb = torch.nn.functional.normalize(pyannote_emb, dim=0)
        except Exception as e:
            print(f"Error extracting pyannote embedding for {filename}: {e}")
            pyannote_emb = torch.zeros(512)

        # Use full audio for style extraction preparation? 
        # We need to crop a segment for style.
        
        audio_tgt = audio.clone()
        audio_style = audio.clone()

        if not self.fine_tuning:
            if self.split:
                # Crop Target Audio
                if audio_tgt.size(1) >= self.segment_size:
                    max_audio_start = audio_tgt.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio_tgt = audio_tgt[:, audio_start:audio_start+self.segment_size]
                else:
                    audio_tgt = torch.nn.functional.pad(audio_tgt, (0, self.segment_size - audio_tgt.size(1)), 'constant')
                
                # Crop Style Audio
                # Logic: We want a random crop of style_segment_size from the SAME file.
                # It can be the same crop or different. Different is better for robustness.
                if audio_style.size(1) >= self.style_segment_size:
                    max_style_start = audio_style.size(1) - self.style_segment_size
                    style_start = random.randint(0, max_style_start)
                    audio_style = audio_style[:, style_start:style_start+self.style_segment_size]
                else:
                    audio_style = torch.nn.functional.pad(audio_style, (0, self.style_segment_size - audio_style.size(1)), 'constant')

            mel = mel_spectrogram(audio_tgt, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            # Fine tuning logic (simplified, assumes we use mels from disk but still need audio for style?)
            # Original code loaded mel from disk.
            # For joint training, we likely aren't in fine_tuning mode usually.
            # But if we are, we need to handle it.
            # Just keep original logic for mel/audio_tgt, and pad audio_style.
            
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio_tgt.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio_tgt = audio_tgt[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio_tgt = torch.nn.functional.pad(audio_tgt, (0, self.segment_size - audio_tgt.size(1)), 'constant')
            
            # For style, we still crop from full audio if possible? 
            # fine_tuning mode assumes cached_wav is available.
            if audio_style.size(1) >= self.style_segment_size:
                max_style_start = audio_style.size(1) - self.style_segment_size
                style_start = random.randint(0, max_style_start)
                audio_style = audio_style[:, style_start:style_start+self.style_segment_size]
            else:
                audio_style = torch.nn.functional.pad(audio_style, (0, self.style_segment_size - audio_style.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio_tgt, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio_tgt.squeeze(0), filename, mel_loss.squeeze(), audio_style.squeeze(0), pyannote_emb)

    def __len__(self):
        return len(self.audio_files)

