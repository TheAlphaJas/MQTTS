import os
import json

# --- Configuration ---
# Adjust these paths to match your system
titw_metadata_path = '/kaggle/input/titw-audio/titw_easy_metadata(1)/bonafide_metadata_cfg_v3/'
titw_audio_path = '/kaggle/input/titw-audio/titw_easy_audio(1)'
output_dir = './kaggle/working/'  # This will be created inside the MQTTS directory

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'audios'), exist_ok=True)


# --- Helper Functions ---
def read_kaldi_file(filepath):
    """Reads a Kaldi-style file and returns a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                data[parts[0]] = parts[1]
    return data

# --- Load Metadata ---
print("Loading TITW metadata files...")
wav_scp = read_kaldi_file(os.path.join(titw_metadata_path, 'wav.scp'))
transcripts = read_kaldi_file(os.path.join(titw_metadata_path, 'text'))
utt2spk = read_kaldi_file(os.path.join(titw_metadata_path, 'utt2spk'))
print(f"Loaded {len(wav_scp)} audio paths, {len(transcripts)} transcripts, and {len(utt2spk)} utterance-to-speaker mappings.")


# --- Process Splits (train, dev) ---
splits_to_process = ['test', 'dev']
for split in splits_to_process:
    print(f"\nProcessing '{split}' split...")
    
    split_audio_dir = os.path.join(titw_audio_path, split)
    if not os.path.exists(split_audio_dir):
        print(f"Warning: Directory not found for split '{split}': {split_audio_dir}. Skipping.")
        continue

    # Find utterance IDs for the current split
    split_utt_ids = {os.path.splitext(filename)[0] for filename in os.listdir(split_audio_dir)}
    print(f"Found {len(split_utt_ids)} utterances in the '{split}' audio directory.")

    json_data = []
    txt_lines = []

    for utt_id in split_utt_ids:
        if utt_id in wav_scp and utt_id in transcripts and utt_id in utt2spk:
            # Create a relative symlink to avoid duplicating audio data
            original_audio_path = wav_scp[utt_id]
            # The wav.scp path might be absolute or relative from a different root.
            # We construct the link based on the audio directory content.
            symlink_path = os.path.join(output_dir, 'audios', f'{utt_id}.wav')
            
            # Use the actual audio file path for the symlink source
            source_audio_path = os.path.join(split_audio_dir, f'{utt_id}.wav')
            
            if not os.path.exists(symlink_path) and os.path.exists(source_audio_path):
                 os.symlink(os.path.abspath(source_audio_path), symlink_path)

            json_entry = {
                'audio_filepath': symlink_path,
                'text': transcripts[utt_id],
                'spk_id': utt2spk[utt_id]
            }
            json_data.append(json_entry)

            # For quantizer training (format: wav_path|text)
            txt_lines.append(f"{symlink_path}|{transcripts[utt_id]}")

    # Write JSON file (for transformer training)
    json_output_path = os.path.join(output_dir, f'{split}.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Successfully created {json_output_path} with {len(json_data)} entries.")

    # Write TXT file (for quantizer training)
    txt_filename = 'training.txt' if split == 'train' else 'validation.txt'
    txt_output_path = os.path.join(output_dir, txt_filename)
    with open(txt_output_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    print(f"Successfully created {txt_output_path} with {len(txt_lines)} lines.")

print("\nData preparation complete.")
