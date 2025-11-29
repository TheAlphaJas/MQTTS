from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tester import Wav2TTS_infer
import argparse
from dp.phonemizer import Phonemizer
import soundfile as sf
import pyloudnorm as pyln
import os
from pathlib import Path
import json
import numpy as np
from collections import Counter
import torch
import dp
torch.serialization.add_safe_globals([dp.preprocessing.text.Preprocessor, dp.preprocessing.text.LanguageTokenizer, dp.preprocessing.text.SequenceTokenizer])


parser = argparse.ArgumentParser()

#Path
parser.add_argument('--phonemizer_dict_path', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--spkr_embedding_path', type=str, default=None)
parser.add_argument('--vocoder_config_path', type=str, required=True)
parser.add_argument('--vocoder_ckpt_path', type=str, required=True)

#Data
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--batch_size', type=int, default=32)

#Sampling
parser.add_argument('--use_repetition_gating', action='store_true')
parser.add_argument('--repetition_penalty', type=float, default=1.0)
parser.add_argument('--sampling_temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--min_top_k', type=int, default=1)
parser.add_argument('--top_p', type=float, default=0.7)
parser.add_argument('--length_penalty_max_length', type=int, default=50)
parser.add_argument('--length_penalty_max_prob', type=float, default=0.8)
parser.add_argument('--max_output_length', type=int, default=100000)
parser.add_argument('--phone_context_window', type=int, default=4)

#Speech Prior
parser.add_argument('--clean_speech_prior', action='store_true')
parser.add_argument('--prior_noise_level', type=float, default=1e-5)
parser.add_argument('--prior_frame', type=int, default=3)

args = parser.parse_args()

args.phoneset = ['<pad>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ',', '.']

with open(args.config_path, 'r') as f:
    argdict = json.load(f)
    assert argdict['sample_rate'] == args.sample_rate, f"Sampling rate not consistent, stated {args.sample_rate}, but the model is trained on {argdict['sample_rate']}"
    # Store command-line args before they get overwritten
    cmdline_args = args.__dict__.copy()
    # Update with config
    args.__dict__.update(argdict)
    # Override with command-line args (excluding None values from defaults)
    for key, value in cmdline_args.items():
        if key in ['config_path', 'model_path', 'input_path', 'outputdir', 'phonemizer_dict_path', 
                   'vocoder_config_path', 'vocoder_ckpt_path', 'spkr_embedding_path', 'batch_size']:
            # These are explicitly set by user, always use them
            args.__dict__[key] = value

if __name__ == '__main__':
    Path(args.outputdir).mkdir(parents=True, exist_ok=True)
    
    # Debug: Print critical paths
    print(f"[INFO] Model path: {args.model_path}")
    print(f"[INFO] Vocoder config: {args.vocoder_config_path}")
    print(f"[INFO] Vocoder checkpoint: {args.vocoder_ckpt_path}")
    print(f"[INFO] Style encoder type: {getattr(args, 'style_encoder_type', None)}")
    
    # Warning: Cannot use pre-computed embeddings with style encoder
    if args.spkr_embedding_path and getattr(args, 'style_encoder_type', None) == 'style_tts2':
        print("\n" + "="*80)
        print("[WARNING] You are using --spkr_embedding_path with a style_tts2 model!")
        print("[WARNING] Pre-computed embeddings only contain Pyannote embeddings.")
        print("[WARNING] The model needs BOTH Pyannote AND StyleTTS2 embeddings (from raw audio).")
        print("[WARNING] Please remove --spkr_embedding_path to process raw audio files instead.")
        print("="*80 + "\n")
        raise ValueError("Cannot use pre-computed speaker embeddings with style encoder model")
    
    meter = pyln.Meter(args.sample_rate)
    phonemizer = Phonemizer.from_checkpoint(args.phonemizer_dict_path)
    with open(args.input_path, 'r') as f:
        input_file = json.load(f)
    model = Wav2TTS_infer(args)
    model.cuda()
    model.vocoder.generator.remove_weight_norm()
    model.vocoder.encoder.remove_weight_norm()
    model.eval()
    i_wavs, i_phones, written = [], [], 0
    for i, (speaker_path, sentence) in enumerate(input_file):
        if args.spkr_embedding_path:
            i_wavs.append(os.path.join(args.spkr_embedding_path, os.path.basename(speaker_path)[:-4] + '.npy'))
        else:
            audio, sr = sf.read(speaker_path)
            assert sr == args.sample_rate
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, -20.0)
            i_wavs.append(audio)
        phones = phonemizer(sentence.strip().lower(), lang='en_us').replace('[', ' ').replace(']', ' ').split()
        phones = [''.join(i for i in phone if not i.isdigit()) for phone in phones if phone.strip()]
        i_phones.append(phones)
        if len(i_wavs) == args.batch_size:
            print (f"Inferencing batch {written//args.batch_size+1}, total {len(input_file)//args.batch_size+1} baches.")
            synthetic = model(i_wavs, i_phones)
            for s in synthetic:
                sf.write(os.path.join(args.outputdir, f'sentence-{written+1}-1.wav'), s, args.sample_rate)
                written += 1
            i_wavs, i_phones = [], []
    if len(i_wavs) > 0:
        synthetic = model(i_wavs, i_phones)
        for s in synthetic:
            sf.write(os.path.join(args.outputdir, f'sentence-{written+1}-1.wav'), s, args.sample_rate)
            written += 1
