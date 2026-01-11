"""
Upload the processed dataset to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --repo-id your-username/agri-foundation-145k --token hf_...

Prerequisites:
    pip install huggingface_hub
"""

import argparse
import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Config
DATASET_DIR = Path("data/release/agri_foundation_v1")

def upload_dataset(repo_id, token=None):
    if not DATASET_DIR.exists():
        print(f"Error: Release directory not found at {DATASET_DIR}")
        print("Run 'scripts/package_for_release.py' first.")
        return

    # Login if token provided
    if token:
        print("Logging in to Hugging Face...")
        login(token=token)
    
    api = HfApi()
    
    print(f"Preparing to upload '{DATASET_DIR}' to '{repo_id}'...")
    
    # Check if repo exists, create if not
    try:
        api.dataset_info(repo_id)
        print(f"Repository '{repo_id}' found.")
    except Exception:
        print(f"Repository '{repo_id}' not found. Creating private dataset...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
            print("Created private repository. You can make it public later in settings.")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return

    # Upload
    print("\nStarting upload... this may take a while for large datasets.")
    print("Files will be uploaded in chunks.")
    
    try:
        url = api.upload_folder(
            folder_path=str(DATASET_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=".",  # Upload to root of the repo
            multi_commits=True,
            multi_commits_verbose=True
        )
        print("\n" + "="*40)
        print("UPLOAD COMPLETE")
        print("="*40)
        print(f"Dataset is live at: {url}")
        print("Don't forget to update the Dataset Card (README.md) on Hugging Face!")
        
    except Exception as e:
        print(f"\nUpload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Repo ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", help="Hugging Face Write Token (optional if already logged in)")
    
    args = parser.parse_args()
    
    upload_dataset(args.repo_id, args.token)
