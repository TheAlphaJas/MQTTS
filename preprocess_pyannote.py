import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")
import torch
import numpy as np
from tqdm import tqdm
from pyannote.audio import Inference
import soundfile as sf
from librosa.util import normalize

def preprocess(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Gather files
    files_to_process = []
    if args.input_training_file and os.path.exists(args.input_training_file):
        with open(args.input_training_file, 'r', encoding='utf-8') as fi:
            files_to_process.extend([os.path.join(args.input_wavs_dir, x.split('|')[0] + '.wav')
                                     for x in fi.read().split('\n') if len(x) > 0])
    # print(files_to_process)
                                     
    if args.input_validation_file and os.path.exists(args.input_validation_file):
        with open(args.input_validation_file, 'r', encoding='utf-8') as fi:
            files_to_process.extend([os.path.join(args.input_wavs_dir, x.split('|')[0] + '.wav')
                                     for x in fi.read().split('\n') if len(x) > 0])
    
    # If no lists provided, verify if we should scan directory? 
    # For now, rely on lists as per other scripts.
    if not files_to_process:
        print("No files found to process. Please check input_training_file/input_validation_file paths.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    # Initialize Pyannote
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        spkr_embedding = Inference("pyannote/embedding", window="whole")
        spkr_embedding.to(device)
    except Exception as e:
        print(f"Failed to initialize Pyannote Inference: {e}")
        print("Make sure you have a valid HF token if required, or use a local model.")
        return

    for filepath in tqdm(files_to_process):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(args.output_dir, filename + '.npy')
        
        if os.path.exists(output_path) and not args.overwrite:
            continue
            
        try:
            # Load audio
            # Pyannote expects tensor (1, T)
            audio, sr = sf.read(filepath)
            # Normalize typically done before embedding extraction in this repo
            audio = normalize(audio) * 0.95
            audio = torch.FloatTensor(audio).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = spkr_embedding({'waveform': audio, 'sample_rate': sr})
            
            # emb is numpy array usually
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
                
            np.save(output_path, emb)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='../../imp_back/datasets/audios')
    parser.add_argument('--input_training_file', default='../../imp_back/datasets/training.txt')
    parser.add_argument('--input_validation_file', default='../../imp_back/datasets/validation.txt')
    parser.add_argument('--output_dir', default='../../imp_back/datasets/speaker_embeddings')
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()
    preprocess(args)

