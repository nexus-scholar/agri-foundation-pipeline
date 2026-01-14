# Agri-Foundation V5 Reproduction Script (Full Reset)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  REPRODUCING DATASET V5 (FULL RESET)        " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 0. Cleanup
Write-Host "`n[0/5] Cleaning up old data..." -ForegroundColor Red
if (Test-Path "data/processed") { Remove-Item "data/processed" -Recurse -Force }
if (Test-Path "data/release") { Remove-Item "data/release" -Recurse -Force }
Write-Host "Cleanup complete."

# 1. Process Raw Data (ALL DATASETS)
Write-Host "`n[1/5] Processing ALL Raw Datasets..." -ForegroundColor Yellow
python process_datasets.py --datasets plantvillage plantdoc tomatoleaf cassava wheat new_plant_diseases plantseg plantwild rocole plantpathology2021
if ($LASTEXITCODE -ne 0) { Write-Error "Processing failed"; exit 1 }

# 2. Verify and Clean
Write-Host "`n[2/5] Verifying and Cleaning..." -ForegroundColor Yellow
python scripts/verify_and_clean_dataset.py
if ($LASTEXITCODE -ne 0) { Write-Error "Verification failed"; exit 1 }

# 3. Package for Release
Write-Host "`n[3/5] Packaging for Release..." -ForegroundColor Yellow
python scripts/package_for_release.py
if ($LASTEXITCODE -ne 0) { Write-Error "Packaging failed"; exit 1 }

# 4. Normalize Class Names
Write-Host "`n[4/5] Normalizing Class Names..." -ForegroundColor Yellow
python scripts/normalize_class_names.py
if ($LASTEXITCODE -ne 0) { Write-Error "Normalization failed"; exit 1 }

# 5. Apply Merges
Write-Host "`n[5/5] Applying Class Merges..." -ForegroundColor Yellow
python scripts/apply_merges.py
if ($LASTEXITCODE -ne 0) { Write-Error "Merging failed"; exit 1 }

Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "  SUCCESS! Dataset V5 is ready in data/release " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
