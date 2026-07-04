# ADA Sample Image Generator - PowerShell Runner
# Run this script to generate synthetic sample images for the change detection POC.
#
# Usage:
#   .\scripts\run_sample_generator.ps1
#   .\scripts\run_sample_generator.ps1 -Count 5 -Size 512
#

param(
    [int]$Count = 3,
    [int]$Size = 512,
    [string]$Out = "data/site/samples",
    [float]$GSD = 0.5
)

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  ADA Sample Image Generator" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $ProjectRoot

Write-Host "`n📂 Project root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "📁 Output:      $Out" -ForegroundColor Yellow
Write-Host "🔢 Samples:     $Count per scenario" -ForegroundColor Yellow
Write-Host "🖼️  Size:       ${Size}x${Size} px" -ForegroundColor Yellow
Write-Host ""

# Check if required packages are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Yellow
$packages = @("numpy", "PIL", "rasterio", "geopandas", "shapely")
$missing = @()

foreach ($pkg in $packages) {
    try {
        python -c "import $pkg" 2>$null
        Write-Host "   ✅ $pkg" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ $pkg" -ForegroundColor Red
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`n⚠️  Missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    $install = Read-Host "Install them now? (y/n)"
    if ($install -eq "y") {
        pip install numpy pillow rasterio geopandas shapely
    } else {
        Write-Host "Please install them manually: pip install numpy pillow rasterio geopandas shapely" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n🚀 Generating sample images..." -ForegroundColor Green
python scripts/create_sample_images.py --out "$Out" --count $Count --size $Size --gsd $GSD

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Done! Sample images created in: $Out" -ForegroundColor Green
    Write-Host ""
    Write-Host "📂 Output structure:" -ForegroundColor Cyan
    Get-ChildItem -Path $Out -Directory | ForEach-Object {
        $scenario = $_.Name
        Write-Host "   📁 $scenario/" -ForegroundColor White
        Get-ChildItem -Path $_.FullName -Directory | ForEach-Object {
            Write-Host "       └── $($_.Name)/" -ForegroundColor Gray
            Get-ChildItem -Path $_.FullName -File | ForEach-Object {
                Write-Host "           ├── $($_.Name)" -ForegroundColor DarkGray
            }
        }
    }
} else {
    Write-Host "❌ Generation failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
