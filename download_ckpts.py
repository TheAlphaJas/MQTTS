import gdown
import os
import shutil

# === Define Google Drive file IDs ===
files = {
    # file_name : (google_drive_file_id, target_directory)
    "config.json": ("1RkzZdSuXzWtSeAKccH45hGPUIusJLsCX", "ckpt/"),
    "g_00600000.ckpt": ("1XOtGWUlem8cG6PCyTyjHHtC5VvUogvlE", "quantizer/checkpoints/"),
    "last.ckpt": ("1JtzeY3kGVks1O1NgwwrYPyPxfDli9kc8", "ckpt/")
}

# === Create target directories if missing ===
for _, (_, target_dir) in files.items():
    os.makedirs(target_dir, exist_ok=True)

# === Download and move files ===
for file_name, (file_id, target_dir) in files.items():
    print(f"Downloading {file_name}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    temp_path = file_name  # downloaded temporarily in current dir
    gdown.download(url, temp_path, quiet=False)

    dest_path = os.path.join(target_dir, file_name)
    shutil.move(temp_path, dest_path)
    print(f"Moved {file_name} → {dest_path}")

print("\n✅ All files downloaded and moved successfully!")
