import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")
import numpy as np
import torch
from tqdm import tqdm
from meldataset import mel_spectrogram, load_wav, MAX_WAV_VALUE
from librosa.util import normalize
import json

def preprocess(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(args.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(args.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(args.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
                            
    all_files = training_files + validation_files
    print(f"Found {len(all_files)} files to process.")
    
    # Load config to get mel params
    with open(args.config) as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for filepath in tqdm(all_files):
        try:
            audio, sampling_rate = load_wav(filepath)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            
            if sampling_rate != config['sampling_rate']:
                # In a real scenario we might resample, but here we just warn/skip or assume data is clean
                pass
                
            audio = torch.FloatTensor(audio).unsqueeze(0).to(device)
            
            mel = mel_spectrogram(audio, config['n_fft'], config['num_mels'],
                                  config['sampling_rate'], config['hop_size'], config['win_size'],
                                  config['fmin'], config['fmax'], center=False)
            
            mel = mel.squeeze(0).cpu().numpy()
            
            filename = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(args.output_dir, filename + '.npy')
            np.save(output_path, mel)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    parser.add_argument('--input_training_file', default='../datasets/training.txt')
    parser.add_argument('--input_validation_file', default='../datasets/validation.txt')
    parser.add_argument('--output_dir', default='../datasets/mels')
    parser.add_argument('--config', default='config.json')
    
    args = parser.parse_args()
    preprocess(args)

