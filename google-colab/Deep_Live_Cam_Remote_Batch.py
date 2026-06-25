# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJ0b2NfdmlzaWJsZSI6dHJ1ZSwiY29sbGFwc2VkX3NlY3Rpb25zIjpbInJ1bnRpbWUtYnVuZGxlIl0sIm5hbWUiOiJEZWVwX0xpdmVfQ2FtX1JlbW90ZV9CYXRjaC5pcHluYiJ9LCJrZXJuZWxzcGVjIjp7ImRpc3BsYXlfbmFtZSI6IlB5dGhvbiAzIiwibGFuZ3VhZ2UiOiJweXRob24iLCJuYW1lIjoicHl0aG9uMyJ9LCJsYW5ndWFnZV9pbmZvIjp7Im5hbWUiOiJweXRob24ifSwibmJmb3JtYXQiOjQsIm5iZm9ybWF0X21pbm9yIjo1fQ==

# %% [markdown] cell=0 id=title
"""MARKDOWN
# Deep-Live-Cam Remote — Colab batch processor

Self-contained, path-based photo/video batch face swap with an optional private Tailscale HTTP/WebSocket controller for the desktop remote app.

## Quick Start

**For batch processing (most common):**
1. Mount Google Drive (section 1 - optional but recommended)
2. Clone repository and install dependencies (section 2)
3. Upload your source face image to: `MyDrive/DeepLiveCamRemote/source/source.png`
4. Upload videos to: `MyDrive/DeepLiveCamRemote/videos/`
5. Configure settings (section 3)
6. Run batch processor (section 5)
7. Download results (section 6)

**For desktop remote app:**
1. Clone repository and install dependencies (section 2) - skip Drive mount
2. Install Tailscale (section 7a)
3. Start API server (section 7b)
4. Use the displayed IP in your desktop remote app

**For photo batches in Colab:**
- Mount Drive (section 1), clone repo (section 2), configure (section 3), then run section 8
"""ENDMARKDOWN

# %% [markdown] cell=1 id=mount-heading
"""MARKDOWN
## 1. Mount Google Drive (OPTIONAL)

⚠️ **Skip this cell if you're only using the Windows remote app** (it handles file transfers via API).

**Run this cell if:**
- You want to batch process videos/photos directly in Colab
- You want persistent storage for outputs

Mount your Google Drive to access source images, videos, and save processed outputs.
"""ENDMARKDOWN

# %% [code] cell=2 id=mount-drive
"""CELL: Mount Google Drive"""
# @title Mount Google Drive
from google.colab import drive
from pathlib import Path
import os

# Check if already mounted
already_mounted = os.path.ismount('/content/drive')

if not already_mounted:
    try:
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
    except Exception as e:
        print(f"Drive mount error: {e}")
        print("If Drive is already mounted, you can continue.")
else:
    print("✓ Google Drive already mounted")

# Verify Drive is accessible
DRIVE_ROOT = Path("/content/drive/MyDrive/DeepLiveCamRemote")
if not Path("/content/drive/MyDrive").exists():
    print("⚠️  WARNING: Google Drive not accessible!")
    print("If you're using the Windows app, you can skip this cell.")
    print("If you need Drive, run: from google.colab import drive; drive.mount('/content/drive', force_remount=True)")
    import sys
    sys.exit(0)  # Exit gracefully instead of crashing

# Create folder structure automatically
folders = [
    DRIVE_ROOT / "source",
    DRIVE_ROOT / "photos",
    DRIVE_ROOT / "videos",
    DRIVE_ROOT / "outputs" / "photos",
    DRIVE_ROOT / "outputs" / "videos",
]

print("\nCreating folder structure...")
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {folder.relative_to(Path('/content/drive/MyDrive'))}")

print("\n" + "="*70)
print("✓ Folder structure ready!")
print("="*70)
print("\nNext steps:")
print("1. Upload your source face image to:")
print("   MyDrive/DeepLiveCamRemote/source/source.png")
print("2. Upload videos to process to:")
print("   MyDrive/DeepLiveCamRemote/videos/")
print("3. Or upload photos to:")
print("   MyDrive/DeepLiveCamRemote/photos/")
print("\nThen run the next cell to install dependencies.")

# %% [markdown] cell=3 id=setup-heading
"""MARKDOWN
## 2. Clone repository and install dependencies

**Note**: This notebook clones the public `djebaz/Deep-Live-Cam-Remote` repository from `main`.
"""ENDMARKDOWN

# %% [code] cell=4 id=setup
"""CELL: Clone and install"""
# @title Clone and install (resumable - safe to re-run after session restart)
import os
import subprocess
import sys
from pathlib import Path
import shutil

REPO_OWNER = "djebaz"
REPO_NAME = "Deep-Live-Cam-Remote"
REPO_BRANCH = "live-webcam-stability"
WORK_DIR = Path("/content/Deep-Live-Cam-Remote")
CRITICAL_FILES = ["colab_api.py", "colab_batch.py", "modules/__init__.py"]

# Clean up Colab sample data (one-time)
if Path("/content/sample_data").exists():
    shutil.rmtree("/content/sample_data")
    print("✓ Removed /content/sample_data")

# Install nvtop if not present
if shutil.which("nvtop") is None:
    subprocess.run(["apt-get", "install", "-qq", "-y", "nvtop"], check=False)
    print("✓ Installed nvtop")
else:
    print("✓ nvtop already installed")

# Create local input/output directories
LOCAL_DIRS = [
    Path("/content/inputs/source"),
    Path("/content/inputs/photos"),
    Path("/content/inputs/videos"),
    Path("/content/outputs/photos"),
    Path("/content/outputs/videos"),
]
for d in LOCAL_DIRS:
    d.mkdir(parents=True, exist_ok=True)
print("✓ Local input/output directories ready")

# Check if repo is already cloned with all critical files
def repo_ready() -> bool:
    if not WORK_DIR.exists():
        return False
    for f in CRITICAL_FILES:
        if not (WORK_DIR / f).exists():
            return False
    return True

if repo_ready():
    print(f"✓ Repository already present: {WORK_DIR}")
    # Pull latest changes
    os.chdir(WORK_DIR)
    result = subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True)
    if result.returncode == 0:
        if "Already up to date" in result.stdout:
            print("✓ Already up to date")
        else:
            print("✓ Pulled latest changes")
            print(result.stdout.strip())
    else:
        print(f"⚠️ git pull failed: {result.stderr.strip()}")
else:
    # Need to clone
    import urllib.request
    import json

    # Check if repo is public
    try:
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            repo_info = json.loads(response.read().decode())
            is_private = repo_info.get("private", True)
    except Exception:
        is_private = True

    # Build clone URL
    if is_private:
        try:
            from google.colab import userdata
            GH_PAT = userdata.get('GH_PAT')
            REPO_URL = f"https://{GH_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
            print("Using authenticated clone (private repo)")
        except Exception as e:
            print(f"Warning: Repository appears private but no GH_PAT available: {e}")
            REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
    else:
        REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
        print("Using public clone (repo is public)")

    os.chdir("/content")

    # Clean up any existing incomplete directories
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    TEMP_CLONE = Path("/content/Deep_Live_Cam_Remote_temp")
    if TEMP_CLONE.exists():
        shutil.rmtree(TEMP_CLONE)

    print(f"Cloning {REPO_OWNER}/{REPO_NAME} (branch: {REPO_BRANCH})...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--branch", REPO_BRANCH, REPO_URL, str(TEMP_CLONE)],
        capture_output=True, text=True, cwd="/content"
    )
    if result.returncode != 0:
        error_msg = result.stderr.replace(REPO_URL, f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git")
        raise RuntimeError(f"Clone failed:\n{error_msg}")

    # Move cloned repository into the stable work directory
    shutil.move(str(TEMP_CLONE), str(WORK_DIR))
    print(f"✓ Repository cloned to: {WORK_DIR}")

# Check if key packages are already installed
def packages_ready() -> bool:
    try:
        import insightface
        import onnxruntime
        import fastapi
        import uvicorn
        return True
    except ImportError:
        return False

if packages_ready():
    print("✓ Python packages already installed")
else:
    print("Installing Python dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy<2",
        "opencv-python==4.10.0.84",
        "insightface==0.7.3",
        "onnx==1.18.0",
        "onnxruntime-gpu==1.23.2",
        "scikit-learn",
        "tqdm",
        "pillow",
        "psutil",
        "protobuf==4.25.1",
        "PySide6>=6.7,<7",
        "cv2_enumerate_cameras==1.1.15",
        "fastapi>=0.115,<1",
        "uvicorn[standard]>=0.30,<1",
        "websockets>=12,<16"
    ], check=True)
    print("✓ Python packages installed")

# Download model if needed
import urllib.request
MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
MODEL_PATH = WORK_DIR / "models" / "inswapper_128.onnx"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024:
    print(f"✓ Model already present: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1048576:.1f} MB)")
else:
    MODEL_PATH.unlink(missing_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".onnx.part")
    temporary_model.unlink(missing_ok=True)
    print("Downloading face swapper model...")
    urllib.request.urlretrieve(MODEL_URL, temporary_model)
    if temporary_model.stat().st_size < 1024 * 1024:
        temporary_model.unlink(missing_ok=True)
        raise RuntimeError("Downloaded inswapper_128.onnx is incomplete")
    temporary_model.replace(MODEL_PATH)
    print(f"✓ Model downloaded: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1048576:.1f} MB)")

# Clean up any cached modules from previous runs
for module_name in list(sys.modules):
    if module_name == "colab_batch" or module_name == "modules" or module_name.startswith("modules."):
        del sys.modules[module_name]

# Set working directory and path
os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

# Show GPU info
subprocess.run(["nvidia-smi"], check=False)

# Verify critical files exist
critical_files = ["colab_api.py", "colab_batch.py", "modules/__init__.py"]
missing_files = [f for f in critical_files if not (WORK_DIR / f).exists()]
if missing_files:
    print(f"ERROR: Missing critical files: {missing_files}")
    print(f"Directory contents: {list(WORK_DIR.glob('*'))}")
    raise RuntimeError(f"Repository clone incomplete. Missing: {missing_files}")

print(f"✓ Runtime ready: {WORK_DIR}")
print(f"✓ Python path: {sys.path[0]}")
print(f"✓ Working directory: {os.getcwd()}")

# %% [markdown] cell=5 id=config-heading
"""MARKDOWN
## 3. Configure Colab paths and processing options
"""ENDMARKDOWN

# %% [code] cell=6 id=config
"""CELL: Batch configuration"""
# @title Batch configuration
DRIVE_ROOT = "/content/drive/MyDrive/DeepLiveCamRemote"
SOURCE_FACE = DRIVE_ROOT + "/source/source.png"
INPUT_DIR = DRIVE_ROOT + "/videos"
PHOTO_INPUT_DIR = DRIVE_ROOT + "/photos"
OUTPUT_DIR = DRIVE_ROOT + "/outputs/videos"
PHOTO_OUTPUT_DIR = DRIVE_ROOT + "/outputs/photos"
ZIP_PATH = DRIVE_ROOT + "/outputs/face_swapped_outputs.zip"
SS = 0.0
DURATION = None  # None processes the remainder
MAX_FPS = 30.0
MAX_WIDTH = 420
MANY_FACES = False
OPACITY = 1.0
SHARPNESS = 0.0
MOUTH_MASK_SIZE = 0.0
POISSON_BLEND = False
COLOR_CORRECTION = False
INTERPOLATION_WEIGHT = 0.0
ENHANCER = "none"  # none, gfpgan, gpen256, gpen512
MAPPING_JSON = None  # e.g. "/content/mapping/face_mapping.json"

# %% [markdown] cell=7 id=mapping-heading
"""MARKDOWN
## 4. Optional: scan identities and edit mapping JSON
Run this before processing only when different target identities need different source faces. Set each generated `source_path`, then set `MAPPING_JSON` above.
"""ENDMARKDOWN

# %% [code] cell=8 id=mapping
"""CELL: Scan identity gallery (optional)"""
# @title Scan identity gallery (optional)
from colab_batch import main
MAPPING_DIR = "/content/mapping"
main(["scan", "--input-dir", INPUT_DIR, "--mapping-dir", MAPPING_DIR])

# %% [markdown] cell=9 id=process-heading
"""MARKDOWN
## 5. Process folder and create ZIP
"""ENDMARKDOWN

# %% [code] cell=10 id=process
"""CELL: Run batch processor"""
# @title Run batch processor
from colab_batch import main
args = ["process", "--input-dir", INPUT_DIR, "--output-dir", OUTPUT_DIR, "--zip-output", ZIP_PATH, "--ss", str(SS), "--max-fps", str(MAX_FPS), "--max-width", str(MAX_WIDTH), "--opacity", str(OPACITY), "--sharpness", str(SHARPNESS), "--mouth-mask-size", str(MOUTH_MASK_SIZE), "--interpolation-weight", str(INTERPOLATION_WEIGHT), "--enhancer", ENHANCER]
if SOURCE_FACE: args += ["--source-face", SOURCE_FACE]
if DURATION is not None: args += ["--duration", str(DURATION)]
if MAPPING_JSON: args += ["--map-config", MAPPING_JSON]
if MANY_FACES: args += ["--many-faces"]
if POISSON_BLEND: args += ["--poisson-blend"]
if COLOR_CORRECTION: args += ["--color-correction"]
exit_code = main(args)
print("Batch exit code:", exit_code)

# %% [markdown] cell=11 id=download-heading
"""MARKDOWN
## 6. Download ZIP
"""ENDMARKDOWN

# %% [code] cell=12 id=download
"""CELL: Download result archive"""
# @title Download result archive
from google.colab import files
files.download(ZIP_PATH)

# %% [markdown] cell=13 id=api-heading
"""MARKDOWN
## 7. Optional: start private Windows app API
Run this after connecting Colab to Tailscale. The Windows app connects to `http://TAILSCALE_IP:7860`.
"""ENDMARKDOWN

# %% [markdown] cell=14 id=tailscale-setup-heading
"""MARKDOWN
### 7a. Install and configure Tailscale

**For automatic auth:** Add `TAILSCALE_AUTHKEY` to Colab secrets (get it from Tailscale admin console > Settings > Keys).

**For manual auth:** If no auth key is set, you'll get a URL to click.

Run these cells to set up secure private networking.
"""ENDMARKDOWN

# %% [code] cell=15 id=tailscale-install
"""CELL: Install Tailscale"""
# @title Install Tailscale (resumable - safe to re-run)
import subprocess
import shutil
import time

def tailscale_installed() -> bool:
    return shutil.which("tailscale") is not None

def daemon_running() -> bool:
    result = subprocess.run(["tailscale", "status"], capture_output=True, text=True)
    # Daemon is running if we get any response (even "Logged out")
    return result.returncode == 0 or "Logged out" in result.stderr or "Logged out" in result.stdout

# Install if needed
if tailscale_installed():
    print("✓ Tailscale already installed")
else:
    print("Installing Tailscale...")
    result = subprocess.run(["curl", "-fsSL", "https://tailscale.com/install.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download Tailscale installer: {result.stderr}")
    install_result = subprocess.run(["sh", "-c", result.stdout], capture_output=True, text=True)
    if install_result.returncode != 0:
        raise RuntimeError(f"Tailscale installation failed: {install_result.stderr}")
    print("✓ Tailscale installed")

# Start daemon if needed
if daemon_running():
    print("✓ Tailscale daemon already running")
else:
    print("Starting Tailscale daemon...")
    daemon_cmd = "sudo tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /dev/null 2>&1 &"
    subprocess.Popen(daemon_cmd, shell=True)
    time.sleep(2)
    if daemon_running():
        print("✓ Tailscale daemon started")
    else:
        print("⚠️  Daemon may not have started correctly - check next cell")

# %% [code] cell=16 id=tailscale-auth
"""CELL: Authenticate Tailscale"""
# @title Start Tailscale and authenticate (resumable)
import subprocess

def get_tailscale_ip() -> str | None:
    result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None

def is_connected() -> bool:
    result = subprocess.run(["tailscale", "status"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    # Connected if we have an IP and status doesn't show logged out
    return get_tailscale_ip() is not None and "Logged out" not in output

# Check if already connected
if is_connected():
    ip = get_tailscale_ip()
    print("="*70)
    print(f"✓ Tailscale already connected! IP: {ip}")
    print("="*70)
    print(f"\nUse this in your Windows app: http://{ip}:7860")
    print("\nRun the 'Start private API server' cell below.")
else:
    # Need to authenticate
    authkey = None
    try:
        from google.colab import userdata
        authkey = userdata.get('TAILSCALE_AUTHKEY')
        print("✓ Found TAILSCALE_AUTHKEY in secrets - using automatic auth")
    except Exception:
        print("⚠️  No TAILSCALE_AUTHKEY found - using interactive auth")

    print("\nConnecting to Tailscale...")
    cmd = ["sudo", "tailscale", "up", "--hostname=colab-dlc-remote"]
    if authkey:
        cmd.append(f"--authkey={authkey}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)

    if result.returncode == 0:
        ip = get_tailscale_ip()
        print("\n" + "="*70)
        print(f"✓ Tailscale connected! IP: {ip}")
        print("="*70)
        print(f"\nUse this in your Windows app: http://{ip}:7860")
        print("\nRun the 'Start private API server' cell below.")
    elif "https://login.tailscale.com" in output:
        print("\n" + "="*70)
        print("IMPORTANT: Click the authentication URL above")
        print("="*70)
        print("\nAfter authorizing, re-run this cell to get your IP.")
    else:
        print("\n⚠️  Authentication may have failed. Check output above.")

# %% [code] cell=17 id=tailscale-ip
"""CELL: Display Tailscale IP"""
# @title Get Tailscale IP address (optional - shown above if connected)
import subprocess

result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True)

if result.returncode == 0 and result.stdout.strip():
    tailscale_ip = result.stdout.strip()
    print("="*70)
    print(f"✓ Your Colab Tailscale IP: {tailscale_ip}")
    print("="*70)
    print(f"\nUse this in your Windows app: http://{tailscale_ip}:7860")
    print("\nNow run the 'Start private API server' cell below.")
else:
    print("Error getting Tailscale IP. Make sure you:")
    print("1. Ran the 'Install Tailscale' cell")
    print("2. Clicked the auth URL in the 'Authenticate Tailscale' cell")
    print("3. Authorized the Colab instance in your Tailscale admin panel")

# %% [markdown] cell=18 id=start-api-heading
"""MARKDOWN
### 7b. Start the API server
Run this cell after Tailscale is configured and you have your IP address.
"""ENDMARKDOWN

# %% [code] cell=19 id=start-api
"""CELL: Start private API server"""
# @title Start private API server
import sys
import threading
from pathlib import Path

# Verify setup was run first
WORK_DIR = Path("/content/Deep-Live-Cam-Remote")
if not WORK_DIR.exists() or str(WORK_DIR) not in sys.path:
    raise RuntimeError(
        "Setup cell not completed successfully! Please run the 'Clone and install' cell first.\n"
        f"Expected directory: {WORK_DIR}\n"
        f"Current sys.path: {sys.path[:3]}"
    )

from colab_api import ensure_drive_layout, app

ensure_drive_layout()

# Run uvicorn in a background thread to avoid asyncio event loop conflicts
def run_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("✓ API server starting on http://0.0.0.0:7860")
print("✓ The server is running in the background")
print("✓ Connect your Tailscale network and use the Tailscale IP with the Windows app")
print("\nPress Ctrl+C or stop the cell to terminate the server.")

# %% [markdown] cell=20 id=photos-heading
"""MARKDOWN
## 8. Optional: run photo batch directly in Colab
"""ENDMARKDOWN

# %% [code] cell=21 id=photos
"""CELL: Run photo batch processor"""
# @title Run photo batch processor
from colab_batch import main
photo_args = ["photos", "--input-dir", PHOTO_INPUT_DIR, "--output-dir", PHOTO_OUTPUT_DIR, "--source-face", SOURCE_FACE, "--opacity", str(OPACITY), "--sharpness", str(SHARPNESS), "--mouth-mask-size", str(MOUTH_MASK_SIZE), "--interpolation-weight", str(INTERPOLATION_WEIGHT), "--enhancer", ENHANCER]
if MANY_FACES: photo_args += ["--many-faces"]
if POISSON_BLEND: photo_args += ["--poisson-blend"]
if COLOR_CORRECTION: photo_args += ["--color-correction"]
exit_code = main(photo_args)
print("Photo batch exit code:", exit_code)
