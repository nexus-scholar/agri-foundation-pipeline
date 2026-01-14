"""
Upload the processed dataset to Zenodo.

Usage:
    python scripts/upload_to_zenodo.py --token YOUR_ZENODO_TOKEN --title "Agri-Foundation-145k Dataset"

Prerequisites:
    1. Get a Zenodo Personal Access Token: https://zenodo.org/account/settings/applications/
    2. Ensure requests and tqdm are installed.
"""

import argparse
import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.zip_utils import zip_folder

# Config
DATASET_DIR = Path("data/release/agri_foundation_v1")
ZIP_PATH = Path("data/release/agri_foundation_v1.zip")
ZENODO_API_URL = "https://zenodo.org/api/deposit/depositions"

class FileWithProgress:
    def __init__(self, path, pbar):
        self.file = open(path, "rb")
        self.pbar = pbar

    def read(self, size=-1):
        chunk = self.file.read(size)
        self.pbar.update(len(chunk))
        return chunk

    def __len__(self):
        return os.fstat(self.file.fileno()).st_size

    def close(self):
        self.file.close()

class ZenodoUploader:
    def __init__(self, token, sandbox=False):
        self.token = token
        self.base_url = "https://sandbox.zenodo.org/api/deposit/depositions" if sandbox else ZENODO_API_URL
        self.params = {'access_token': token}

    def create_deposition(self):
        print("Creating new deposition...")
        response = requests.post(self.base_url, params=self.params, json={}, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()

    def upload_file(self, deposition_id, bucket_url, file_path):
        file_name = file_path.name
        upload_url = f"{bucket_url}/{file_name}"
        
        print(f"Uploading {file_name} to Zenodo...")
        file_size = file_path.stat().st_size
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
            wrapped_file = FileWithProgress(file_path, pbar)
            try:
                response = requests.put(
                    upload_url,
                    data=wrapped_file,
                    params=self.params
                )
            finally:
                wrapped_file.close()
        
        response.raise_for_status()
        print(f"\nSuccessfully uploaded {file_name}")
        return response.json()

    def update_metadata(self, deposition_id, title, description, creators=None):
        print("Updating deposition metadata...")
        if creators is None:
            creators = [{'name': 'Anonymous', 'affiliation': 'Research'}]
        
        data = {
            'metadata': {
                'title': title,
                'upload_type': 'dataset',
                'description': description,
                'creators': creators,
                'access_right': 'open',
                'license': 'CC-BY-4.0'
            }
        }
        
        url = f"{self.base_url}/{deposition_id}"
        response = requests.put(url, params=self.params, data=json.dumps(data), headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()

def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Zenodo")
    parser.add_argument("--token", required=True, help="Zenodo Personal Access Token")
    parser.add_argument("--title", default="Agri-Foundation-145k Dataset", help="Title of the deposition")
    parser.add_argument("--description", default="A unified dataset for agricultural foundation models, containing 145k+ images across multiple crop diseases.", help="Description of the dataset")
    parser.add_argument("--creator", help="Name of the creator (e.g., 'John Doe')")
    parser.add_argument("--affiliation", default="Research", help="Affiliation of the creator")
    parser.add_argument("--sandbox", action="store_true", help="Use Zenodo Sandbox (for testing)")
    parser.add_argument("--skip-zip", action="store_true", help="Skip zipping if the zip file already exists")
    parser.add_argument("--deposition-id", help="ID of an existing deposition to update")
    
    args = parser.parse_args()

    # 1. Zip the folder (Only if we are NOT just updating metadata or if we plan to upload)
    # If deposition_id is provided, we assume we might only want to update metadata, 
    # UNLESS the user explicitly asks to upload/zip? 
    # For now, let's keep it simple: if dep_id is provided, we skip zipping/uploading 
    # unless we add a flag later to force re-upload.
    # But wait, the user might want to re-upload the file to an existing deposition?
    # Let's assume if dep_id is present, we skip file operations for now as per request.
    
    if not args.deposition_id:
        if not DATASET_DIR.exists():
            print(f"Error: Release directory not found at {DATASET_DIR}")
            print("Run 'scripts/package_for_release.py' first.")
            return

        # 1. Zip the folder
        if not ZIP_PATH.exists() or not args.skip_zip:
            zip_folder(DATASET_DIR, ZIP_PATH)
        else:
            print(f"Using existing zip at {ZIP_PATH}")

    # 2. Interact with Zenodo
    uploader = ZenodoUploader(args.token, sandbox=args.sandbox)
    
    try:
        if args.deposition_id:
            dep_id = args.deposition_id
            print(f"Updating existing deposition: {dep_id}")
            # We don't upload files in this mode for safety/simplicity unless requested
        else:
            deposition = uploader.create_deposition()
            dep_id = deposition['id']
            bucket_url = deposition['links']['bucket']
            
            print(f"Deposition created with ID: {dep_id}")
            
            uploader.upload_file(dep_id, bucket_url, ZIP_PATH)
        
        creators = None
        if args.creator:
            creators = [{'name': args.creator, 'affiliation': args.affiliation}]
            
        uploader.update_metadata(dep_id, args.title, args.description, creators=creators)
        
        print("\n" + "="*40)
        print("UPDATE COMPLETE" if args.deposition_id else "UPLOAD COMPLETE")
        print("="*40)
        print(f"Deposition ID: {dep_id}")
        base_url = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
        print(f"View/Edit at: {base_url}/deposit/{dep_id}")
        print("\nNOTE: You still need to manually click 'Publish' in the Zenodo UI after reviewing.")
        
    except Exception as e:
        print(f"\nUpload failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    main()
