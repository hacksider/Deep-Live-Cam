<h1 align="center">Deep-Live-Cam 2.1</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## Disclaimer

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- Ethical Use: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.

- Content Restrictions: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.

- Legal Compliance: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.

- User Responsibility: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## Exclusive v2.7 beta Quick Start - Pre-built (Windows/Mac Silicon/CPU)

<a href="https://deeplivecam.net/index.php/quickstart"> <img src="media/Download.png" width="285" height="77" />

##### This is the fastest build you can get if you have a discrete NVIDIA or AMD GPU, CPU or Mac Silicon, And you'll receive special priority support. 2.7 beta is the best you can have with 30+ extra features than the open source version.

###### These Pre-builts are perfect for non-technical users or those who don't have time to, or can't manually install all the requirements. Just a heads-up: this is an open-source project, so you can also install it manually.

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

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the quickstart version.**

<details>
<summary>Click to see the process</summary>

### Installation

This setup follows the current code and `requirements.txt`.

**1. Prerequisites**

- Python 3.9+ (3.11 recommended)
- pip
- git
- `ffmpeg` and `ffprobe` available in PATH (required for video extraction/encoding)
- GUI runtime (`tkinter`) available in your Python installation
- [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Dependency notes from `requirements.txt`:

- `insightface==0.7.3`
- `opencv-python==4.10.0.84`
- `customtkinter==5.2.2`
- `pygrabber` (Windows webcam device listing)
- `tensorflow` on non-macOS systems
- `onnxruntime-gpu` on non-macOS systems
- `onnxruntime-silicon` on Apple Silicon macOS

**2. Clone**

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**3. Create a virtual environment and install dependencies**

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Models and how they are loaded**

Place manual models in the `models` folder.

| Model                     | Path                             | Required               | Download                                    | Notes                                                                                  |
| ------------------------- | -------------------------------- | ---------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `inswapper_128_fp16.onnx` | `models/inswapper_128_fp16.onnx` | Recommended            | Auto on first run (face swapper pre-check)  | Preferred swapper model when present.                                                  |
| `inswapper_128.onnx`      | `models/inswapper_128.onnx`      | Optional fallback      | Manual                                      | Used as fallback when fp16 model is not available.                                     |
| `gfpgan-1024.onnx`        | `models/gfpgan-1024.onnx`        | Optional               | Manual                                      | Required only when `face_enhancer` is enabled.                                         |
| `GFPGANv1.4.onnx`         | `models/GFPGANv1.4.onnx`         | Optional (legacy file) | Manual                                      | Current enhancer code loads `gfpgan-1024.onnx`; keep this as an optional legacy asset. |
| `buffalo_l`               | InsightFace model cache          | Required for swapping  | Auto by InsightFace at first initialization | Needed for detection/recognition; must be available for source/target face analysis.   |

**5. Run**

```bash
python run.py
```

This starts CPU mode by default.

### Execution Modes

- CPU (default): `python run.py`
- CUDA: `python run.py --execution-provider cuda`
- CoreML (Apple Silicon): `python run.py --execution-provider coreml`
- DirectML (Windows): `python run.py --execution-provider directml`
- OpenVINO (Intel): `python run.py --execution-provider openvino`

Model selection behavior in runtime:

- If `models/inswapper_128_fp16.onnx` exists, it is used first.
- If fp16 model is missing and `models/inswapper_128.onnx` exists, runtime falls back to it.
- You do not need both swapper files to run.

### Video and Live Mode Notes

- Video processing requires `ffmpeg` and `ffprobe` in PATH.
- Webcam mode requires camera access permissions.
- On Windows, webcam device names are read through `pygrabber` / DirectShow.
- If no source face or target faces are detected, swapping is skipped and status logs are shown.
</details>

## Usage

**1. Image/Video Mode**

- Execute `python run.py`.
- Choose a source face image and a target image/video.
- Click "Start".
- The output will be saved in a directory named after the target video.

**2. Webcam Mode**

- Execute `python run.py`.
- Select a source face image.
- Confirm your camera appears in the camera dropdown (Windows uses DirectShow via `pygrabber`).
- Click "Live".
- Wait for the preview to appear (10-30 seconds).
- Use a screen capture tool like OBS to stream.
- To change the face, select a new source image.

## Download all models in this huggingface link

- [**Download models here**](https://huggingface.co/hacksider/deep-live-cam/tree/main)

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

- [**Ars Technica**](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - _"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"_
- [**Yahoo!**](https://www.yahoo.com/tech/ok-viral-ai-live-stream-080041056.html) - _"OK, this viral AI live stream software is truly terrifying"_
- [**CNN Brasil**](https://www.cnnbrasil.com.br/tecnologia/ia-consegue-clonar-rostos-na-webcam-entenda-funcionamento/) - _"AI can clone faces on webcam; understand how it works"_
- [**Bloomberg Technoz**](https://www.bloombergtechnoz.com/detail-news/71032/kenalan-dengan-teknologi-deep-live-cam-bisa-jadi-alat-menipu) - _"Get to know Deep Live Cam technology, it can be used as a tool for deception."_
- [**TrendMicro**](https://www.trendmicro.com/vinfo/gb/security/news/cyber-attacks/ai-vs-ai-deepfakes-and-ekyc) - _"AI vs AI: DeepFakes and eKYC"_
- [**PetaPixel**](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - _"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"_
- [**SomeOrdinaryGamers**](https://www.youtube.com/watch?time*continue=1074&v=py4Tc-Y8BcY) - *"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"\_
- [**IShowSpeed**](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - _"Alright look look look, now look chat, we can do any face we want to look like chat"_
- [**TechLinked (Linus Tech Tips)**](https://www.youtube.com/watch?v=wnCghLjqv3s&t=551s) - _"They do a pretty good job matching poses, expression and even the lighting"_
- [**IShowSpeed**](https://youtu.be/JbUPRmXRUtE?t=3964) - \*"What the F***! Why do I look like Vinny Jr? I look exactly like Vinny Jr!? No, this shit is crazy! Bro This is F*** Crazy!"\*

## Credits

- [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
- [Henry](https://github.com/henryruhs): One of the major contributor in this repo
- [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
- [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
- [GosuDRM](https://github.com/GosuDRM): for the open version of roop
- [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
- [vic4key](https://github.com/vic4key): For supporting/contributing to this project
- [kier007](https://github.com/kier007): for improving the user experience
- [qitianai](https://github.com/qitianai): for multi-lingual support
- [laurigates](https://github.com/laurigates): Decoupling stuffs to make everything faster!
- and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
- Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
- All the wonderful users who helped make this project go viral by starring the repo ❤️

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
