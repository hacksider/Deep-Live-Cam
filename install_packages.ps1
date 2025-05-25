# Define a local cache directory
$CacheDir = "$HOME\pip_cache"
if (!(Test-Path $CacheDir)) { New-Item -ItemType Directory -Path $CacheDir }

# List of package URLs
$Packages = @(
    "https://download.pytorch.org/whl/cu118/torch-2.5.1%2Bcu118-cp38-cp38-win_amd64.whl"
    "https://download.pytorch.org/whl/cu118/torchvision-0.20.1-cp38-cp38-win_amd64.whl"
    "https://files.pythonhosted.org/packages/.../numpy-1.23.5.whl"
    "https://files.pythonhosted.org/packages/.../onnx-1.16.0.whl"
    # Add other package URLs here
)

# Function to download using wget or Invoke-WebRequest
Function Download-Package {
    param([string]$url, [string]$dest)
    if (Get-Command wget -ErrorAction SilentlyContinue) {
        wget -c $url -O $dest
    } else {
        Invoke-WebRequest -Uri $url -OutFile $dest
    }
}

# Download packages with resume support
foreach ($url in $Packages) {
    $FileName = Split-Path -Path $url -Leaf
    $DestPath = "$CacheDir\$FileName"
    Download-Package -url $url -dest $DestPath
}

# Install packages from the cache
pip install --find-links="$CacheDir" numpy typing-extensions opencv-python `
cv2_enumerate_cameras onnx insightface psutil tk customtkinter pillow `
torch torchvision onnxruntime-gpu tensorflow opennsfw2 protobuf