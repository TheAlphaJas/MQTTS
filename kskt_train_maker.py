import json
import os

def extract_filename(path):
    """Extract the audio filename from a path."""
    return os.path.basename(path)

def generate_filtered_json(map_json_path, main_json_path, output_json_path):
    # Load map json
    with open(map_json_path, "r") as f:
        map_data = json.load(f)

    # Extract all filenames from the map json list of [path, text]
    map_filenames = set(extract_filename(item[0]) for item in map_data)

    print(f"Found {len(map_filenames)} audio files in map JSON.")

    # Load main json
    with open(main_json_path, "r") as f:
        main_data = json.load(f)

    print(f"Main JSON contains {len(main_data)} entries.")

    # Filter only those entries whose key is in map_filenames
    filtered = {k: v for k, v in main_data.items() if k in map_filenames}

    print(f"Filtered JSON contains {len(filtered)} matched audio files.")

    # Write out the filtered json
    with open(output_json_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved filtered JSON to {output_json_path}")


if __name__ == "__main__":
    # Example usage
    generate_filtered_json(
        map_json_path="./TITW_Finaltest.json",
        main_json_path="../../imp_back/Testout/test_q.json",
        output_json_path="../../imp_back/Testout/test_q_kskt.json"
    )
