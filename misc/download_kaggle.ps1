<#
.SYNOPSIS
    Downloads a Kaggle dataset ZIP and extracts it to a local folder.
    
.USAGE
    ./download_kaggle.ps1 <KAGGLE_URL> <FOLDER_NAME> [LOCAL_PATH]

.EXAMPLE
    ./download_kaggle.ps1 "https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/download" "chest-xrays" "D:\Datasets"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$KaggleUrl,

    [Parameter(Mandatory = $true)]
    [string]$FolderName,

    [string]$LocalPath = "$env:USERPROFILE\Downloads"
)

# --- Prepare paths ---
$ZipFile = Join-Path $env:TEMP "$FolderName.zip"
$DestinationFolder = Join-Path $LocalPath $FolderName

Write-Host "Downloading dataset from:" $KaggleUrl
Write-Host "Saving ZIP as:" $ZipFile

# --- Download dataset ZIP ---
try {
    Invoke-WebRequest -Uri $KaggleUrl -OutFile $ZipFile -UseBasicParsing
    Write-Host "✅ Download complete."
} catch {
    Write-Host "❌ Failed to download dataset. Check Kaggle URL or login requirements."
    exit 1
}

# --- Create destination folder ---
if (-Not (Test-Path -Path $DestinationFolder)) {
    New-Item -ItemType Directory -Path $DestinationFolder | Out-Null
    Write-Host "📁 Created folder:" $DestinationFolder
} else {
    Write-Host "📁 Folder already exists:" $DestinationFolder
}

# --- Extract ZIP ---
Write-Host "Extracting files..."
Expand-Archive -LiteralPath $ZipFile -DestinationPath $DestinationFolder -Force
Write-Host "✅ Extraction complete."

# --- Cleanup ZIP ---
Remove-Item -Path $ZipFile -Force
Write-Host "🧹 Cleaned up temporary ZIP file."

Write-Host "🎯 Files available at:" $DestinationFolder
Write-Host "🚀 Done!"