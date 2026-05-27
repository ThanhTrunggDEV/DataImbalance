import os
import subprocess

def download_kaggle_dataset(dataset_name="shashwatwork/knee-osteoarthritis-dataset-with-severity", download_path="./data"):
    """
    Download and unzip kaggle dataset. 
    Requires kaggle API token to be placed in ~/.kaggle/kaggle.json
    """
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading dataset {dataset_name}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path, "--unzip"], check=True)
        print("Download and unzip complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure your kaggle.json is configured correctly.")

if __name__ == "__main__":
    download_kaggle_dataset()