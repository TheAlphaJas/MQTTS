import argparse
import json
import os
from tqdm import tqdm

def update_json_with_text(args):
    # 1. Load the existing JSON metadata
    print(f"Loading existing metadata from {args.metadata_path}...")
    with open(args.metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} entries in metadata.")

    # 2. Read the original text file list (e.g., training.txt)
    # Format is expected to be: filename|text|speaker|...
    # We need to map filename -> text
    text_map = {}
    
    files_to_read = []
    if args.input_file_list:
        files_to_read.append(args.input_file_list)
    
    # If the user provided separate training and validation lists, we can read both to build a comprehensive map
    # But typically this script runs once for training.json and once for validation.json
    
    print(f"Reading text mappings from {args.input_file_list}...")
    with open(args.input_file_list, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[1]
                text_map[filename] = text
    
    print(f"Loaded {len(text_map)} text entries.")

    # 3. Update metadata
    updated_count = 0
    missing_count = 0
    
    for key in tqdm(metadata.keys()):
        # The key in metadata is usually the filename (without extension, or relative path)
        # Let's check how it matches with text_map keys.
        # MQTTS preprocess often uses the basename as key.
        
        # Try direct match
        if key in text_map:
            metadata[key]['text'] = text_map[key]
            updated_count += 1
        else:
            # Try matching just the basename if key is a path
            basename = os.path.basename(key)
            # Or removing extension if present
            basename_no_ext = os.path.splitext(basename)[0]
            
            if basename in text_map:
                metadata[key]['text'] = text_map[basename]
                updated_count += 1
            elif basename_no_ext in text_map:
                metadata[key]['text'] = text_map[basename_no_ext]
                updated_count += 1
            else:
                # Sometimes text_map keys are full paths?
                # Let's try to match text_map keys against the current metadata key
                found = False
                for t_key in text_map:
                    if t_key.endswith(key) or key.endswith(t_key):
                        metadata[key]['text'] = text_map[t_key]
                        updated_count += 1
                        found = True
                        break
                
                if not found:
                    missing_count += 1
                    # print(f"Warning: No text found for {key}")

    print(f"Updated {updated_count} entries.")
    if missing_count > 0:
        print(f"Warning: {missing_count} entries were missing text.")

    # 4. Save updated metadata
    output_path = args.output_path if args.output_path else args.metadata_path
    print(f"Saving updated metadata to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=True, help="Path to the existing .json metadata file (e.g., datasets/training.json)")
    parser.add_argument('--input_file_list', type=str, required=True, help="Path to the original text file list (e.g., datasets/training.txt)")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the updated json. If None, overwrites input.")
    
    args = parser.parse_args()
    update_json_with_text(args)

