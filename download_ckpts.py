import gdown
import os
import shutil
import subprocess
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub")
    exit()

# === Define Google Drive file IDs and target dirs ===
# Format: "filename": ("DRIVE_ID", "TARGET_DIR")
gdrive_files = {
    "config.json": ("1RkzZdSuXzWtSeAKccH45hGPUIusJLsCX", "ckpt/"),
    "g_00600000.ckpt": ("1XOtGWUlem8cG6PCyTyjHHtC5VvUogvlE", "quantizer/checkpoints/"),
    "last.ckpt": ("1JtzeY3kGVks1O1NgwwrYPyPxfDli9kc8", "ckpt/"),
    "g_00036500": ("1RbLJZPRtfTLwqOb5WGcO1ZnEZoRhB8Cs", "quantizer/checkpoints/"),
    # Add MQTTS pretrained placeholder
    "pretrained_mqtts.ckpt": ("1rfXoqJ4Wb3Yj_ig4xKsluEF1Ugm9L0pf", "ckpt/") # Placeholder ID
}

# === Define Hugging Face files ===
# Format: "filename": ("REPO_ID", "REMOTE_FILENAME", "TARGET_DIR")
hf_files = {
    "epoch_2nd_00020.pth": ("yl4579/StyleTTS2-LibriTTS", "epoch_2nd_00020.pth", "ckpt/")
}

# === Create directories if missing ===
for _, (_, target_dir) in gdrive_files.items():
    os.makedirs(target_dir, exist_ok=True)
for _, (_, _, target_dir) in hf_files.items():
    os.makedirs(target_dir, exist_ok=True)

# === Download GDrive files ===
print("\n=== Downloading Google Drive Files ===")
for file_name, (file_id, target_dir) in gdrive_files.items():
    dest_path = os.path.join(target_dir, file_name)
    if os.path.exists(dest_path):
        print(f"✅ {file_name} already exists, skipping download.")
        continue
    
    if not file_id:
        print(f"⚠️  Skipping {file_name}: No Google Drive ID provided.")
        continue

    print(f"⬇️  Downloading {file_name}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    temp_path = file_name
    gdown.download(url, temp_path, quiet=False)
    if os.path.exists(temp_path):
        shutil.move(temp_path, dest_path)
        print(f"📦 Moved {file_name} → {dest_path}")
    else:
        print(f"❌ Failed to download {file_name}")

# === Download Hugging Face files ===
print("\n=== Downloading Hugging Face Files ===")
for file_name, (repo_id, remote_filename, target_dir) in hf_files.items():
    dest_path = os.path.join(target_dir, file_name)
    if os.path.exists(dest_path):
        print(f"✅ {file_name} already exists, skipping download.")
        continue

    print(f"⬇️  Downloading {file_name} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=remote_filename)
        shutil.copy(downloaded_path, dest_path) # hf_hub_download caches it, so we copy to target
        print(f"📦 Copied {file_name} → {dest_path}")
    except Exception as e:
        print(f"❌ Failed to download {file_name}: {e}")

print("\n✅ All files processed!")
