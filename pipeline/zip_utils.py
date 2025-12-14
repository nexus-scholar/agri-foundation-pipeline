"""Zip extraction utilities with Windows long-path support."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from .fs_utils import copytree, count_files, ensure_dir, safe_rmtree


def unzip_dataset(zip_path: Path, extract_to: Path, dataset_name: str):
    """Unzip a dataset into `extract_to` handling long Windows paths."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    print("\n" + "=" * 60)
    print(f"UNZIPPING {dataset_name.upper()}")
    print("=" * 60)
    if not zip_path.exists():
        print(f"ERROR: Zip file not found: {zip_path}")
        return None

    temp_extract = extract_to.parent / f"_temp_extract_{dataset_name}"
    if extract_to.exists():
        print(f"Removing existing folder: {extract_to}")
        safe_rmtree(extract_to)
    if temp_extract.exists():
        safe_rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    renamed_files: list[dict[str, str]] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            print(f"Extracting {len(members)} files...")
            root_folders = {m.split("/")[0] for m in members if "/" in m}
            print(f"Root folder(s) in zip: {root_folders}")

            for member in members:
                if member.endswith("/"):
                    continue
                member_path = Path(member)
                target_path = temp_extract / member
                ensure_dir(target_path.parent)
                try:
                    data = zip_ref.read(member)
                    with open(target_path, "wb") as fh:
                        fh.write(data)
                except Exception:
                    data = zip_ref.read(member)
                    short_name, short_path = _shorten_filename(member_path, target_path)
                    with open(short_path, "wb") as fh:
                        fh.write(data)
                    renamed_files.append({
                        "original": str(member_path),
                        "short_name": short_name,
                        "relative_parent": str(member_path.parent)
                    })
        print(f"Extracted {count_files(temp_extract)} files, skipped 0")

        extracted_items = list(temp_extract.iterdir())
        print(f"Found {len(extracted_items)} items in temp folder")
        folders_to_copy = _find_folders_to_copy(extracted_items, temp_extract)
        if not folders_to_copy:
            print("ERROR: No content found to extract")
            safe_rmtree(temp_extract)
            return None

        source_total = sum(count_files(f) for f in folders_to_copy)
        print(f"Total source files to copy: {source_total}")

        print("  Copying to destination...")
        extract_to.mkdir(parents=True, exist_ok=True)

        for folder in folders_to_copy:
            print(f"  Copying {folder.name}...")
            # Copy contents of each folder into the destination
            dest_subfolder = extract_to / folder.name
            copytree(folder, dest_subfolder)

        copied_total = count_files(extract_to)
        print(f"  Copy successful ({copied_total} files)")
        if copied_total != source_total:
            print(f"  WARNING: File count mismatch (source={source_total}, dest={copied_total})")

        print("Cleaning up temp folder...")
        safe_rmtree(temp_extract)

        if not extract_to.exists():
            print("ERROR: Could not find extracted content")
            return None

        if renamed_files:
            manifest = extract_to.parent / f"{dataset_name}_renamed_files.json"
            with open(manifest, "w", encoding="utf-8") as fh:
                json.dump(renamed_files, fh, indent=2)
            print(f"Recorded {len(renamed_files)} renamed files in {manifest}")
        print(f"Successfully extracted to {extract_to} ({copied_total} files)")
        return extract_to
    except zipfile.BadZipFile:
        print(f"ERROR: Invalid zip file: {zip_path}")
        return None
    except Exception as exc:  # pragma: no cover - top-level logging handles
        print(f"ERROR: Failed to extract {zip_path}: {exc}")
        return None


def _shorten_filename(member_path: Path, target_path: Path):
    original_name = member_path.name
    ext = member_path.suffix
    base_hash = hash(original_name) % 1_000_000
    counter = 0
    while True:
        suffix = f"_{counter}" if counter else ""
        short_name = f"img_{base_hash:06d}{suffix}{ext}"
        short_path = target_path.parent / short_name
        if not short_path.exists():
            return short_name, short_path
        counter += 1


def _find_folders_to_copy(extracted_items: list[Path], temp_extract: Path) -> list[Path]:
    """Find all folders to copy, including case-variant duplicates."""
    folder_info = []
    for item in extracted_items:
        if item.is_dir():
            file_count = count_files(item)
            print(f"  - {item.name}: {file_count} files")
            folder_info.append((item, file_count, item.name.lower()))

    if not folder_info:
        total = count_files(temp_extract)
        if total > 0:
            print(f"Using temp folder directly: {total} files")
            return [temp_extract]
        return []

    # Check for case-insensitive duplicates (e.g., PlantVillage and plantvillage)
    lower_names = {}
    for item, count, lower_name in folder_info:
        if lower_name not in lower_names:
            lower_names[lower_name] = []
        lower_names[lower_name].append((item, count))

    folders_to_copy = []

    # If we have case duplicates, include all of them
    for lower_name, items in lower_names.items():
        if len(items) > 1:
            total_files = sum(count for _, count in items)
            names = [item.name for item, _ in items]
            print(f"  [INFO] Found case-variant folders: {names} (total {total_files} files)")
            print(f"  All variants will be copied and merged during processing")
            # Add all case variants
            for item, _ in items:
                folders_to_copy.append(item)
        else:
            # Single folder, just add it
            folders_to_copy.append(items[0][0])

    return folders_to_copy
