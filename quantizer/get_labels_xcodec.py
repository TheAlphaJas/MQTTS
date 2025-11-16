"""
Generate quantization labels using XCodec instead of the original quantizer.
This replaces get_labels.py for XCodec integration.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import numpy as np
import argparse
import json
import torch
from tqdm import tqdm
from librosa.util import normalize
from modules.xcodec_wrapper import XCodecWrapper
import soundfile as sf


def load_wav(full_path):
    """Load audio file."""
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate


def inference(a):
    """Run inference to generate quantization codes."""
    print("Initializing XCodec model...")
    xcodec = XCodecWrapper(
        model_name=a.xcodec_model_name,
        n_code_groups=a.n_code_groups,
        sample_rate=a.sample_rate,
        freeze_encoder=True  # Use pretrained weights
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xcodec = xcodec.to(device)
    xcodec.eval()
    
    # Load metadata
    with open(a.input_json, 'r') as f:
        t_file = json.load(f)
    filelist = [os.path.join(a.input_wav_dir, l) for l in t_file]
    
    print(f"Processing {len(filelist)} files...")
    
    with torch.no_grad():
        for filename in tqdm(list(t_file.keys())):
            fname = os.path.join(a.input_wav_dir, filename)
            
            # Load and normalize audio
            audio, sampling_rate = load_wav(fname)
            if sampling_rate != a.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=a.sample_rate)
            
            audio = audio / np.max(np.abs(audio))  # Normalize
            audio = normalize(audio) * 0.95
            
            # Convert to tensor
            x = torch.FloatTensor(audio).to(device)
            
            # Encode with XCodec
            z_q, loss_q, codes = xcodec(x.unsqueeze(0))
            
            # Convert codes to list format compatible with MQTTS
            # codes is a list of tensors, each of shape (B*T,)
            # We need to reshape to (T, n_code_groups)
            batch_size = 1
            if len(codes) > 0:
                # Get temporal dimension
                T = codes[0].size(0) // batch_size
                # Stack codes
                codes_tensor = torch.stack(codes, dim=-1)  # (B*T, n_code_groups)
                codes_tensor = codes_tensor.reshape(batch_size, T, -1)  # (B, T, n_code_groups)
                codes_list = codes_tensor[0].cpu().numpy().T.tolist()  # (n_code_groups, T) -> list
            else:
                codes_list = []
            
            # Store in metadata format
            t_file[filename]['quantization'] = codes_list
    
    # Save updated metadata
    with open(a.output_json, 'w') as f:
        json.dump(t_file, f, indent=2)
    
    print(f"Saved quantization codes to {a.output_json}")


def main():
    print('Initializing XCodec Inference Process..')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../datasets/train.json')
    parser.add_argument('--input_wav_dir', default='../datasets/audios')
    parser.add_argument('--output_json', default='../datasets/train_q.json')
    parser.add_argument('--xcodec_model_name', type=str, default='facebook/xcodec-base',
                        help='HuggingFace model name for XCodec')
    parser.add_argument('--n_code_groups', type=int, default=4,
                        help='Number of code groups (must match MQTTS config)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    
    a = parser.parse_args()
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    inference(a)


if __name__ == '__main__':
    main()

