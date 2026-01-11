"""Data processing utilities."""
from __future__ import annotations

import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional

def calculate_md5(file_path: Path) -> Optional[str]:
    """
    Calculate MD5 hash of a file for deduplication.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Hex digest string or None if error.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with error handling.
    
    Args:
        path: Path to CSV file.
        
    Returns:
        pd.DataFrame or empty DataFrame if not found/error.
    """
    if not path.exists():
        print(f"Error: File not found at {path}")
        return pd.DataFrame()
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()
