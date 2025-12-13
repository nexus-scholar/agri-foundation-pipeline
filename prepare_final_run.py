#!/usr/bin/env python3
"""
Prepare for Final Experiment Run

This script:
1. Archives old experiment results (run with incorrect 8-class tomato mapping)
2. Cleans up the experiments directory
3. Verifies data configuration is correct

Run this BEFORE the final experiment run.

Usage:
    python prepare_final_run.py
    python prepare_final_run.py --keep-old  # Don't archive, just verify
"""
import shutil
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results/experiments")
ARCHIVE_DIR = Path("results/archived_experiments")


def archive_old_experiments():
    """Move old experiments to archive folder."""
    if not RESULTS_DIR.exists():
        print("[OK] No experiments directory found.")
        return 0
    
    # Get all experiment folders
    exp_folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir()]
    json_files = [f for f in RESULTS_DIR.iterdir() if f.suffix == '.json']
    
    if not exp_folders and not json_files:
        print("[OK] No experiments to archive.")
        return 0
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"pre_final_run_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[ARCHIVE] Moving {len(exp_folders)} experiment folders to {archive_path}")
    
    # Move folders
    for folder in exp_folders:
        dest = archive_path / folder.name
        shutil.move(str(folder), str(dest))
        print(f"  Moved: {folder.name}")
    
    # Move JSON files
    for jf in json_files:
        dest = archive_path / jf.name
        shutil.move(str(jf), str(dest))
        print(f"  Moved: {jf.name}")
    
    print(f"\n[OK] Archived {len(exp_folders)} folders and {len(json_files)} JSON files")
    return len(exp_folders)


def verify_configuration():
    """Verify data configuration is correct."""
    print("\n" + "="*60)
    print("CONFIGURATION VERIFICATION")
    print("="*60)
    
    try:
        from src.config.crop_configs import CROP_CONFIGS, get_crop_config
        
        all_good = True
        
        for crop_name, config in CROP_CONFIGS.items():
            print(f"\n[{crop_name.upper()}]")
            print(f"  Canonical classes: {config.num_classes}")
            print(f"  Source-only (PDA): {config.source_only_classes or 'None'}")
            print(f"  Is Partial Domain: {config.is_partial_domain}")
            
            # Check specific requirements
            if crop_name == "tomato":
                if config.num_classes != 9:
                    print(f"  [ERROR] Expected 9 classes, got {config.num_classes}")
                    all_good = False
                if "tomato_spider_mites" not in config.canonical_classes:
                    print(f"  [ERROR] Spider mites not in canonical classes!")
                    all_good = False
                else:
                    print(f"  [OK] Spider mites included")
                    
            elif crop_name == "potato":
                if config.num_classes != 3:
                    print(f"  [ERROR] Expected 3 classes, got {config.num_classes}")
                    all_good = False
                if "potato_healthy" not in config.source_only_classes:
                    print(f"  [WARN] potato_healthy should be source-only for PDA")
                    
            elif crop_name == "pepper":
                if config.num_classes != 2:
                    print(f"  [ERROR] Expected 2 classes, got {config.num_classes}")
                    all_good = False
        
        if all_good:
            print("\n[OK] All configurations verified!")
        else:
            print("\n[ERROR] Configuration issues found!")
            
        return all_good
        
    except ImportError as e:
        print(f"[ERROR] Could not import config: {e}")
        return False


def print_final_checklist():
    """Print final checklist for the run."""
    print("\n" + "="*60)
    print("FINAL RUN CHECKLIST")
    print("="*60)
    print("""
Before running experiments on Colab:

1. [x] Tomato: 9 classes (spider_mites included)
2. [x] Potato: 3 classes (healthy = PDA scenario)  
3. [x] Pepper: 2 classes (full alignment)
4. [x] Old experiments archived
5. [ ] Upload latest code to Colab
6. [ ] Run: python colab_experiment_runner.py --all

Expected output:
- Phase 1: 12 baseline experiments
- Phase 2: 3 strong augmentation experiments
- Phase 3: 4 AL strategy experiments  
- Phase 4: 3 FixMatch experiments
- Phase 5: 4 architecture benchmark experiments
- Total: 26 experiments

Estimated time: ~4-6 hours on Colab Pro (T4/V100)
""")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-old', action='store_true', 
                        help="Don't archive old experiments")
    args = parser.parse_args()
    
    print("="*60)
    print("PREPARING FOR FINAL EXPERIMENT RUN")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Archive old experiments
    if not args.keep_old:
        archived = archive_old_experiments()
    else:
        print("[SKIP] Keeping old experiments (--keep-old)")
    
    # Step 2: Verify configuration
    config_ok = verify_configuration()
    
    # Step 3: Print checklist
    print_final_checklist()
    
    if config_ok:
        print("\n" + "="*60)
        print("[READY] You are ready for the FINAL experiment run!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[STOP] Fix configuration issues before running!")
        print("="*60)


if __name__ == "__main__":
    main()

