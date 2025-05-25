<h1 align="center">Deep-Live-Cam</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

##  Disclaimer

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- Ethical Use: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.

- Content Restrictions: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.

- Legal Compliance: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.

- User Responsibility: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## Exclusive v2.0 Quick Start - Pre-built (Windows)

  <a href="https://deeplivecam.net/index.php/quickstart"> <img src="media/Download.png" width="285" height="77" />

##### This is the fastest build you can get if you have a discrete NVIDIA or AMD GPU.
 
###### These Pre-builts are perfect for non-technical users or those who don't have time to, or can't manually install all the requirements. Just a heads-up: this is an open-source project, so you can also install it manually. This will be 60 days ahead on the open source version.

## TLDR; Live Deepfake in just 3 Clicks
![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)
1. Select a face
2. Select which camera to use
3. Press live!

## Features & Uses - Everything is in real-time

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450"> 
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

### Omegle

**Surprise people on Omegle**

<p align="center">
  <video src="https://github.com/user-attachments/assets/2e9b9b82-fa04-4b70-9f56-b1f68e7672d0" width="450" controls></video>
</p>

## Installation (Manual)

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the prebuilt version.**

<details>
<summary>Click to see the process</summary>

### Installation

This is more likely to work on your computer but will be slower as it utilizes the CPU.

**1. Set up Your Platform**

-   Python (3.10 recommended)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) - ```iex (irm ffmpeg.tc.ht)```
-   [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**2. Clone the Repository**

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**3. Download the Models**

1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. [inswapper\_128\_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies**

We highly recommend using a `venv` to avoid issues.


**For Windows:**

It is highly recommended to use Python 3.10 for Windows for best compatibility with all features and dependencies.

**Automated Setup (Recommended):**

1.  **Run the setup script:**
    Double-click `setup_windows.bat` or run it from your command prompt:
    ```batch
    setup_windows.bat
    ```
    This script will:
    *   Check if Python is in your PATH.
    *   Warn if `ffmpeg` is not found (see "Manual Steps / Notes" below for ffmpeg help).
    *   Create a virtual environment named `.venv` (consistent with macOS setup).
    *   Activate the virtual environment for the script's session.
    *   Upgrade pip.
    *   Install Python packages from `requirements.txt`.
    Wait for the script to complete. It will pause at the end; press any key to close the window if you double-clicked it.

2.  **Run the application:**
    After setup, use the provided `.bat` scripts to run the application. These scripts automatically activate the correct virtual environment:
    *   `run_windows.bat`: Runs the application with the CPU execution provider by default. This is a good starting point if you don't have a dedicated GPU or are unsure.
    *   `run-cuda.bat`: Runs with the CUDA (NVIDIA GPU) execution provider. Requires an NVIDIA GPU and CUDA Toolkit installed (see GPU Acceleration section).
    *   `run-directml.bat`: Runs with the DirectML (AMD/Intel GPU on Windows) execution provider.

    Example: Double-click `run_windows.bat` to launch the UI, or run from a command prompt:
    ```batch
    run_windows.bat --source path\to\your_face.jpg --target path\to\video.mp4
    ```

**Manual Steps / Notes:**

*   **Python:** Ensure Python 3.10 is installed and added to your system's PATH. You can download it from [python.org](https://www.python.org/downloads/).
*   **ffmpeg:**
    *   `ffmpeg` is required for video processing. The `setup_windows.bat` script will warn if it's not found in your PATH.
    *   An easy way to install `ffmpeg` on Windows is to open PowerShell as Administrator and run:
        ```powershell
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); choco install ffmpeg -y
        ```
        Alternatively, download from [ffmpeg.org](https://ffmpeg.org/download.html), extract the files, and add the `bin` folder (containing `ffmpeg.exe`) to your system's PATH environment variable. The original README also linked to a [YouTube guide](https://www.youtube.com/watch?v=OlNWCpFdVMA) or `iex (irm ffmpeg.tc.ht)` via PowerShell.
*   **Visual Studio Runtimes:** If you encounter errors during `pip install` for packages that compile C code (e.g., some scientific computing or image processing libraries), you might need the [Visual Studio Build Tools (or Runtimes)](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Ensure "C++ build tools" (or similar workload) are selected during installation.
*   **Virtual Environment (Manual Alternative):** If you prefer to set up the virtual environment manually instead of using `setup_windows.bat`:
    ```batch
    python -m venv .venv 
    .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```
    (The new automated scripts use `.venv` as the folder name for consistency with the macOS setup).

For Linux:
```bash
# Ensure you use the installed Python 3.10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For macOS:**

For a streamlined setup on macOS, use the provided shell scripts:

1.  **Make scripts executable:**
    Open your terminal, navigate to the cloned `Deep-Live-Cam` directory, and run:
    ```bash
    chmod +x setup_mac.sh
    chmod +x run_mac*.sh
    ```

2.  **Run the setup script:**
    This will check for Python 3.9+, ffmpeg, create a virtual environment (`.venv`), and install required Python packages.
    ```bash
    ./setup_mac.sh
    ```
    If you encounter issues with specific packages during `pip install` (especially for libraries that compile C code, like some image processing libraries), you might need to install system libraries via Homebrew (e.g., `brew install jpeg libtiff ...`) or ensure Xcode Command Line Tools are installed (`xcode-select --install`).

3.  **Activate the virtual environment (for manual runs):**
    After setup, if you want to run commands manually or use developer tools from your terminal session:
    ```bash
    source .venv/bin/activate
    ```
    (To deactivate, simply type `deactivate` in the terminal.)

4.  **Run the application:**
    Use the provided run scripts for convenience. These scripts automatically activate the virtual environment.
    *   `./run_mac.sh`: Runs the application with the CPU execution provider by default. This is a good starting point.
    *   `./run_mac_cpu.sh`: Explicitly uses the CPU execution provider.
    *   `./run_mac_coreml.sh`: Attempts to use the CoreML execution provider for potential hardware acceleration on Apple Silicon and Intel Macs.
    *   `./run_mac_mps.sh`: Attempts to use the MPS (Metal Performance Shaders) execution provider, primarily for Apple Silicon Macs.

    Example of running with specific source/target arguments:
    ```bash
    ./run_mac.sh --source path/to/your_face.jpg --target path/to/video.mp4
    ```
    Or, to simply launch the UI:
    ```bash
    ./run_mac.sh
    ```

**Important Notes for macOS GPU Acceleration (CoreML/MPS):**

*   The `setup_mac.sh` script installs packages from `requirements.txt`, which typically includes a general CPU-based version of `onnxruntime`.
*   For optimal performance on Apple Silicon (M1/M2/M3) or specific GPU acceleration, you might need to install a different `onnxruntime` package *after* running `setup_mac.sh` and while the virtual environment (`.venv`) is active.
*   **Example for `onnxruntime-silicon` (often requires Python 3.10 for older versions like 1.13.1):**
    The original `README` noted that `onnxruntime-silicon==1.13.1` was specific to Python 3.10. If you intend to use this exact version for CoreML:
    ```bash
    # Ensure you are using Python 3.10 if required by your chosen onnxruntime-silicon version
    # After running setup_mac.sh and activating .venv:
    # source .venv/bin/activate
    
    pip uninstall onnxruntime onnxruntime-gpu # Uninstall any existing onnxruntime
    pip install onnxruntime-silicon==1.13.1   # Or your desired version
    
    # Then use ./run_mac_coreml.sh
    ```
    Check the ONNX Runtime documentation for the latest recommended packages for Apple Silicon.
*   **For MPS with ONNX Runtime:** This may require a specific build or version of `onnxruntime`. Consult the ONNX Runtime documentation. For PyTorch-based operations (like the Face Enhancer or Hair Segmenter if they were PyTorch native and not ONNX), PyTorch should automatically try to use MPS on compatible Apple Silicon hardware if available.
*   **User Interface (Tkinter):** If you encounter errors related to `_tkinter` not being found when launching the UI, ensure your Python installation supports Tk. For Python installed via Homebrew, this is usually `python-tk` (e.g., `brew install python-tk@3.9` or `brew install python-tk@3.10`, matching your Python version).

** In case something goes wrong and you need to reinstall the virtual environment **

```bash
# Deactivate the virtual environment
rm -rf venv

# Reinstall the virtual environment
python -m venv venv
source venv/bin/activate

# install the dependencies again
pip install -r requirements.txt
```

**Run:** If you don't have a GPU, you can run Deep-Live-Cam using `python run.py`. Note that initial execution will download models (~300MB).

### GPU Acceleration

**CUDA Execution Provider (Nvidia)**

1. Install [CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

3. Usage:

```bash
python run.py --execution-provider cuda
```

**CoreML Execution Provider (Apple Silicon)**

Apple Silicon (M1/M2/M3) specific installation:

1. Make sure you've completed the macOS setup above using Python 3.10.
2. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.13.1
```

3. Usage (important: specify Python 3.10):

```bash
python3.10 run.py --execution-provider coreml
```

**Important Notes for macOS:**
- You **must** use Python 3.10, not newer versions like 3.11 or 3.13
- Always run with `python3.10` command not just `python` if you have multiple Python versions installed
- If you get error about `_tkinter` missing, reinstall the tkinter package: `brew reinstall python-tk@3.10`
- If you get model loading errors, check that your models are in the correct folder
- If you encounter conflicts with other Python versions, consider uninstalling them:
  ```bash
  # List all installed Python versions
  brew list | grep python
  
  # Uninstall conflicting versions if needed
  brew uninstall --ignore-dependencies python@3.11 python@3.13
  
  # Keep only Python 3.10
  brew cleanup
  ```

**CoreML Execution Provider (Apple Legacy)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.13.1
```

2. Usage:

```bash
python run.py --execution-provider coreml
```

**DirectML Execution Provider (Windows)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.15.1
```

2. Usage:

```bash
python run.py --execution-provider directml
```

**OpenVINO™ Execution Provider (Intel)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.15.0
```

2. Usage:

```bash
python run.py --execution-provider openvino
```
</details>

## Usage

**1. Image/Video Mode**

-   Execute `python run.py`.
-   Choose a source face image and a target image/video.
-   Click "Start".
-   The output will be saved in a directory named after the target video.

**2. Webcam Mode**

-   Execute `python run.py`.
-   Select a source face image.
-   Click "Live".
-   Wait for the preview to appear (10-30 seconds).
-   Use a screen capture tool like OBS to stream.
-   To change the face, select a new source image.

## Tips and Tricks

Check out these helpful guides to get the most out of Deep-Live-Cam:

- [Unlocking the Secrets to the Perfect Deepfake Image](https://deeplivecam.net/index.php/blog/tips-and-tricks/unlocking-the-secrets-to-the-perfect-deepfake-image) - Learn how to create the best deepfake with full head coverage
- [Video Call with DeepLiveCam](https://deeplivecam.net/index.php/blog/tips-and-tricks/video-call-with-deeplivecam) - Make your meetings livelier by using DeepLiveCam with OBS and meeting software
- [Have a Special Guest!](https://deeplivecam.net/index.php/blog/tips-and-tricks/have-a-special-guest) - Tutorial on how to use face mapping to add special guests to your stream
- [Watch Deepfake Movies in Realtime](https://deeplivecam.net/index.php/blog/tips-and-tricks/watch-deepfake-movies-in-realtime) - See yourself star in any video without processing the video
- [Better Quality without Sacrificing Speed](https://deeplivecam.net/index.php/blog/tips-and-tricks/better-quality-without-sacrificing-speed) - Tips for achieving better results without impacting performance
- [Instant Vtuber!](https://deeplivecam.net/index.php/blog/tips-and-tricks/instant-vtuber) - Create a new persona/vtuber easily using Metahuman Creator

Visit our [official blog](https://deeplivecam.net/index.php/blog/tips-and-tricks) for more tips and tutorials.

## Command Line Arguments (Unmaintained)

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            the live camera display as you see it in the front-facing camera frame
  --live-resizable                                         the live camera frame is resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```

Looking for a CLI mode? Using the -s/--source argument will make the run program in cli mode.

## Press

**We are always open to criticism and are ready to improve, that's why we didn't cherry-pick anything.**

 - [*"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"*](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - Ars Technica
 - [*"Thanks Deep Live Cam, shapeshifters are among us now"*](https://dataconomy.com/2024/08/15/what-is-deep-live-cam-github-deepfake/) - Dataconomy
 - [*"This free AI tool lets you become anyone during video-calls"*](https://www.newsbytesapp.com/news/science/deep-live-cam-ai-impersonation-tool-goes-viral/story) - NewsBytes
 - [*"OK, this viral AI live stream software is truly terrifying"*](https://www.creativebloq.com/ai/ok-this-viral-ai-live-stream-software-is-truly-terrifying) - Creative Bloq
 - [*"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - PetaPixel
 - [*"Deep-Live-Cam Uses AI to Transform Your Face in Real-Time, Celebrities Included"*](https://www.techeblog.com/deep-live-cam-ai-transform-face/) - TechEBlog
 - [*"An AI tool that "makes you look like anyone" during a video call is going viral online"*](https://telegrafi.com/en/a-tool-that-makes-you-look-like-anyone-during-a-video-call-is-going-viral-on-the-Internet/) - Telegrafi
 - [*"This Deepfake Tool Turning Images Into Livestreams is Topping the GitHub Charts"*](https://decrypt.co/244565/this-deepfake-tool-turning-images-into-livestreams-is-topping-the-github-charts) - Emerge
 - [*"New Real-Time Face-Swapping AI Allows Anyone to Mimic Famous Faces"*](https://www.digitalmusicnews.com/2024/08/15/face-swapping-ai-real-time-mimic/) - Digital Music News
 - [*"This real-time webcam deepfake tool raises alarms about the future of identity theft"*](https://www.diyphotography.net/this-real-time-webcam-deepfake-tool-raises-alarms-about-the-future-of-identity-theft/) - DIYPhotography
 - [*"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - SomeOrdinaryGamers
 - [*"Alright look look look, now look chat, we can do any face we want to look like chat"*](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - IShowSpeed

## Credits

-   [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
-   [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
-   [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
-   [GosuDRM](https://github.com/GosuDRM): for the open version of roop
-   [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
-   [vic4key](https://github.com/vic4key): For supporting/contributing to this project
-   [kier007](https://github.com/kier007): for improving the user experience
-   [qitianai](https://github.com/qitianai): for multi-lingual support
-   and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
-   Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
-   All the wonderful users who helped make this project go viral by starring the repo ❤️

[![Stargazers](https://reporoster.com/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

## Stars to the Moon 🚀

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
