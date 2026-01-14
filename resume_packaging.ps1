# Resume V5 Reproduction (From Packaging)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  RESUMING DATASET V5 (PACKAGING & MERGE)    " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 3. Package for Release
Write-Host "`n[3/5] Packaging for Release (using Hard Links)..." -ForegroundColor Yellow
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
