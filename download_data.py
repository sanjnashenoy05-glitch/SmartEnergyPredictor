"""
download_data.py
Download and extract the UCI Individual Household Electric Power Consumption dataset.
Official UCI URL:
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
Backup URL:
https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
"""

import os
import sys
import zipfile
import requests

UCI_PRIMARY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
UCI_BACKUP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
EXPECTED_FILENAME = "household_power_consumption.txt"


def download_dataset(data_dir="data", force=False):
    """
    Downloads and extracts the UCI Household Electric Power Consumption dataset.
    Returns the path to household_power_consumption.txt.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_txt = os.path.join(data_dir, EXPECTED_FILENAME)
    target_zip = os.path.join(data_dir, "household_power_consumption.zip")

    if os.path.exists(target_txt) and not force:
        size_mb = os.path.getsize(target_txt) / (1024 * 1024)
        print(f"Dataset already exists at {target_txt} ({size_mb:.1f} MB). Skipping download.")
        return target_txt

    urls = [UCI_PRIMARY_URL, UCI_BACKUP_URL]
    download_success = False

    for url in urls:
        print(f"Attempting download from: {url}")
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code != 200:
                print(f"Received status code {r.status_code} from {url}")
                continue

            content_len = r.headers.get("content-length")
            total_size = int(content_len) if content_len else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            with open(target_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            pct = min(100.0, (downloaded / total_size) * 100.0)
                            sys.stdout.write(f"\rDownloading: {pct:5.1f}% [{downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
                        else:
                            sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB downloaded...")
                        sys.stdout.flush()

            print("\nDownload complete.")
            download_success = True
            break
        except Exception as e:
            print(f"\nFailed to download from {url}: {e}")
            if os.path.exists(target_zip):
                os.remove(target_zip)

    if not download_success:
        raise RuntimeError(
            f"Could not download dataset from UCI mirrors. Please manually download the zip from "
            f"{UCI_PRIMARY_URL} or {UCI_BACKUP_URL} and extract '{EXPECTED_FILENAME}' into '{data_dir}/'."
        )

    print(f"Extracting {target_zip} to {data_dir}...")
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    if not os.path.exists(target_txt):
        raise FileNotFoundError(f"Extraction finished but {target_txt} not found.")

    final_size_mb = os.path.getsize(target_txt) / (1024 * 1024)
    print(f"Successfully extracted {target_txt} ({final_size_mb:.1f} MB).")
    return target_txt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download UCI Household Power Consumption dataset")
    parser.add_argument("--data-dir", default="data", help="Directory to save dataset")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    args = parser.parse_args()

    download_dataset(data_dir=args.data_dir, force=args.force)
