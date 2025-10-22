import os
import zipfile
import gdown

def download_and_unzip(file_url, output_filename):
    """
    Downloads a file from a Google Drive URL using gdown and extracts it if it's a zip file.

    Args:
        file_url (str): The Google Drive URL of the file to download.
        output_filename (str): The name to save the downloaded file as.
    """
    # 1. Download the file using gdown
    print(f"Downloading {output_filename}...")
    try:
        # Using fuzzy=True helps gdown parse various Google Drive link formats
        gdown.download(file_url, output=output_filename, quiet=False, fuzzy=True)
        print(f"Successfully downloaded {output_filename}.")
    except Exception as e:
        print(f"Failed to download {output_filename}. Error: {e}")
        return

    # 2. Extract the zip file
    if output_filename.endswith('.zip'):
        # Define the directory for extraction (e.g., "file.zip" -> "file/")
        extract_path = os.path.splitext(output_filename)[0]
        
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        print(f"Extracting {output_filename} to '{extract_path}/'...")
        with zipfile.ZipFile(output_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction of {output_filename} complete.")

if __name__ == '__main__':
    # A list of tuples, where each tuple contains the URL and desired filename
    files_to_process = [
        ("https://drive.google.com/uc?id=1xCNSj2E7BGIFPgviEU-H6mIpA_wb5IOp", 'titw_easy_metadata.zip'),
        ("https://drive.google.com/uc?id=1_ezdqBAw6SaeVBF9R-K_63lzlTsUnXux", 'titw_easy_audio.zip')
    ]

    for url, filename in files_to_process:
        print("-" * 50)
        download_and_unzip(url, filename)

    print("\nAll tasks are complete.")

