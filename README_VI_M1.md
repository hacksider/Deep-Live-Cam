# 🚀 Deep-Live-Cam - Hướng Dẫn Tối Ưu Cho MacBook M1 Pro Max

Hướng dẫn chi tiết triển khai và tối ưu Deep-Live-Cam cho **MacBook M1 Pro Max** với 64GB RAM và 32 GPU cores.

---

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Cài Đặt Nhanh](#cài-đặt-nhanh)
- [Các Script Có Sẵn](#các-script-có-sẵn)
- [Hướng Dẫn Sử Dụng Chi Tiết](#hướng-dẫn-sử-dụng-chi-tiết)
- [So Sánh Các Chế Độ](#so-sánh-các-chế-độ)
- [Tối Ưu Hiệu Năng](#tối-ưu-hiệu-năng)
- [Xử Lý Lỗi](#xử-lý-lỗi)
- [Tips & Tricks](#tips--tricks)

---

## 🎯 Giới Thiệu

Deep-Live-Cam là công cụ hoán đổi khuôn mặt realtime và deepfake video. Với MacBook M1 Pro Max của bạn, bạn có thể đạt hiệu năng tối đa nhờ:

- ✅ **Apple Neural Engine** (32 GPU cores)
- ✅ **64GB Unified Memory** - chia sẻ giữa CPU và GPU
- ✅ **10-core CPU** với hiệu năng cao
- ✅ **CoreML optimization** cho Apple Silicon

### 🎬 Hiệu Năng Dự Kiến

| Tác Vụ | FPS | Chất Lượng |
|--------|-----|------------|
| Webcam 720p | 25-35 FPS | Cao |
| Webcam 1080p | 18-25 FPS | Cao |
| Video 1080p (Speed) | 30-45 FPS | Tốt |
| Video 1080p (Balanced) | 20-30 FPS | Rất tốt |
| Video 1080p (Quality) | 15-25 FPS | Xuất sắc |
| Video 4K | 8-15 FPS | Xuất sắc |

---

## 🚀 Cài Đặt Nhanh

### Bước 1: Clone Repository

```bash
cd ~/Documents
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

### Bước 2: Chạy Script Setup

```bash
chmod +x setup-m1-pro-max.sh
./setup-m1-pro-max.sh
```

Script này sẽ tự động:
- ✅ Kiểm tra và cài đặt Homebrew
- ✅ Cài đặt Python 3.10, ffmpeg, python-tk
- ✅ Tạo virtual environment
- ✅ Cài đặt dependencies
- ✅ Tối ưu hóa với onnxruntime-silicon
- ✅ Tải models (GFPGANv1.4.pth và inswapper_128.onnx)
- ✅ Cấp quyền thực thi cho các script

### Bước 3: Sẵn Sàng Sử Dụng!

```bash
./start-webcam.sh     # Chế độ webcam
./start-balanced.sh   # Chế độ cân bằng (khuyến nghị)
./start-quality.sh    # Chất lượng cao nhất
./start-speed.sh      # Tốc độ cao nhất
```

---

## 📂 Các Script Có Sẵn

### 1. 🎥 `start-webcam.sh` - Chế độ Webcam Realtime

**Khi nào dùng:** Streaming trực tiếp, video call, live stream

**Cấu hình:**
```
- Memory: 48 GB
- Threads: 10
- Processors: Face Swapper
- Features: Many faces, Resizable, Mirror
```

**Cách dùng:**
```bash
./start-webcam.sh
```

**Hiệu năng:** 25-35 FPS (720p), 18-25 FPS (1080p)

---

### 2. ⚡ `start-speed.sh` - Chế độ Tốc Độ Cao

**Khi nào dùng:** Xử lý video dài, cần kết quả nhanh

**Cấu hình:**
```
- Memory: 40 GB
- Threads: 12
- Processors: Face Swapper only
- Encoder: H.264 (fast)
- Quality: 20
```

**Cách dùng:**
```bash
# Với GUI
./start-speed.sh

# Command line (nhanh hơn)
./start-speed.sh source.jpg input_video.mp4 output.mp4
```

**Hiệu năng:** 30-45 FPS (1080p)

---

### 3. ⚖️ `start-balanced.sh` - Chế độ Cân Bằng (Khuyến nghị)

**Khi nào dùng:** Sử dụng hàng ngày, cân bằng tốc độ và chất lượng

**Cấu hình:**
```
- Memory: 48 GB
- Threads: 10
- Processors: Face Swapper
- Encoder: H.264
- Quality: 12
```

**Cách dùng:**
```bash
# Với GUI
./start-balanced.sh

# Command line
./start-balanced.sh source.jpg input_video.mp4 output.mp4
```

**Hiệu năng:** 20-30 FPS (1080p)

---

### 4. 💎 `start-quality.sh` - Chế độ Chất Lượng Cao Nhất

**Khi nào dùng:** Video quan trọng, cần chất lượng tốt nhất

**Cấu hình:**
```
- Memory: 56 GB
- Threads: 8
- Processors: Face Swapper + Face Enhancer
- Encoder: H.265 (HEVC)
- Quality: 4 (rất cao)
```

**Cách dùng:**
```bash
# Với GUI
./start-quality.sh

# Command line
./start-quality.sh source.jpg input_video.mp4 output.mp4
```

**Hiệu năng:** 15-25 FPS (1080p), 8-15 FPS (4K)

---

## 📖 Hướng Dẫn Sử Dụng Chi Tiết

### 🎥 A. Sử Dụng Chế độ Webcam

1. **Chạy script:**
   ```bash
   ./start-webcam.sh
   ```

2. **Trong GUI:**
   - Click "Select a face" → chọn ảnh khuôn mặt nguồn
   - Click nút "Live"
   - Cho phép truy cập camera khi macOS hỏi
   - Đợi 10-30 giây để preview xuất hiện

3. **Để stream:**
   - Mở OBS Studio
   - Thêm source → Window Capture
   - Chọn cửa sổ Deep-Live-Cam
   - Stream như bình thường

4. **Để đổi khuôn mặt:**
   - Click "Select a face" → chọn ảnh mới
   - Khuôn mặt sẽ tự động thay đổi

---

### 🎬 B. Xử Lý Video

#### Cách 1: Sử Dụng GUI

```bash
./start-balanced.sh  # hoặc quality/speed
```

Trong GUI:
1. "Select a face" → chọn ảnh khuôn mặt nguồn
2. "Select a target" → chọn video cần xử lý
3. Click "Start"
4. Đợi xử lý hoàn tất
5. Video output sẽ ở cùng thư mục với tên mới

#### Cách 2: Command Line (Nhanh hơn)

```bash
# Chất lượng cao
./start-quality.sh my_face.jpg input.mp4 output_quality.mp4

# Cân bằng
./start-balanced.sh my_face.jpg input.mp4 output_balanced.mp4

# Tốc độ cao
./start-speed.sh my_face.jpg input.mp4 output_fast.mp4
```

---

### 🎭 C. Xử Lý Nhiều Khuôn Mặt (Face Mapping)

```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Chạy với face mapping
python run.py \
  --source face1.jpg \
  --target video_with_multiple_people.mp4 \
  --output output.mp4 \
  --execution-provider coreml \
  --max-memory 56 \
  --map-faces \
  --many-faces \
  --frame-processor face_swapper face_enhancer \
  --keep-fps \
  --keep-audio
```

**Trong GUI với Face Mapping:**
1. Chạy script với flag `--map-faces`
2. Select nhiều source faces
3. Map từng face với target face trong video
4. Process như bình thường

---

## 📊 So Sánh Các Chế Độ

### Bảng So Sánh

| Tính Năng | Speed | Balanced | Quality | Webcam |
|-----------|-------|----------|---------|---------|
| **Tốc độ** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Chất lượng** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RAM Usage** | 40 GB | 48 GB | 56 GB | 48 GB |
| **CPU Threads** | 12 | 10 | 8 | 10 |
| **Face Enhancer** | ❌ | ❌ | ✅ | ❌ |
| **Best For** | Long videos | Daily use | Important | Streaming |

### Khi Nào Dùng Chế Độ Nào?

- **🎥 Webcam:** Streaming, video calls, live demos
- **⚡ Speed:** Video dài (>10 phút), nhiều videos cần xử lý
- **⚖️ Balanced:** ⭐ Sử dụng hàng ngày, video thông thường
- **💎 Quality:** Video quan trọng, presentation, portfolio

---

## ⚡ Tối Ưu Hiệu Năng

### 1. Tối Ưu macOS

```bash
# Tắt App Nap cho Terminal
defaults write NSGlobalDomain NSAppSleepDisabled -bool YES

# Enable High Performance Mode (cần restart)
sudo nvram boot-args="serverperfmode=1 $(nvram boot-args 2>/dev/null | cut -f 2-)"

# Kiểm tra
nvram boot-args
```

### 2. Monitor Hiệu Năng

#### Dùng Activity Monitor
1. Mở Activity Monitor
2. Window → GPU History
3. Window → CPU History
4. Xem Memory pressure

#### Dùng Terminal

```bash
# Cài htop
brew install htop

# Chạy monitoring
htop

# Hoặc dùng top
top -o cpu
```

### 3. Cooling & Thermal Management

MacBook M1 Pro Max có thể nóng khi xử lý nặng:

**Giải pháp:**
- Dùng đế tản nhiệt
- Đặt máy ở nơi thoáng mát, không phủ vải
- Nếu quá nóng, giảm `--execution-threads`:
  ```bash
  # Thay vì 10-12 threads, dùng 6-8
  python run.py --execution-threads 6 ...
  ```

### 4. Tùy Chỉnh Memory Allocation

**Nếu bạn mở nhiều app khác:**
```bash
# Giảm max-memory xuống
./start-balanced.sh  # Sửa max-memory từ 48 → 32
```

**Nếu chỉ chạy Deep-Live-Cam:**
```bash
# Tăng lên tối đa
./start-quality.sh  # max-memory 56-60 GB
```

### 5. Batch Processing (Xử Lý Nhiều Videos)

```bash
#!/bin/bash
# batch-process.sh

source venv/bin/activate

FACE="my_face.jpg"
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

for video in input_videos/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Processing: $filename"

    python run.py \
      --source "$FACE" \
      --target "$video" \
      --output "$OUTPUT_DIR/${filename}_swapped.mp4" \
      --execution-provider coreml \
      --max-memory 48 \
      --execution-threads 10 \
      --frame-processor face_swapper \
      --keep-fps \
      --keep-audio \
      --video-quality 12

    echo "Completed: $filename"
done

echo "All videos processed!"
```

Chạy:
```bash
chmod +x batch-process.sh
./batch-process.sh
```

---

## 🔧 Xử Lý Lỗi

### ❌ Lỗi: "Could not find onnxruntime-silicon"

```bash
source venv/bin/activate
pip uninstall -y onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.16.3
```

### ❌ Lỗi: "tkinter module not found"

```bash
brew reinstall python-tk@3.10
```

### ❌ Lỗi: "Camera not accessible"

1. System Settings → Privacy & Security → Camera
2. Bật quyền cho Terminal hoặc app bạn đang dùng
3. Restart Terminal

### ❌ Lỗi: "ModuleNotFoundError: No module named 'cv2'"

```bash
source venv/bin/activate
pip install opencv-python==4.8.1.78
```

### ❌ Lỗi: Models không tải được

```bash
# Tải thủ công
mkdir -p models

# GFPGANv1.4
curl -L -o models/GFPGANv1.4.pth \
  "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"

# inswapper_128
curl -L -o models/inswapper_128.onnx \
  "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
```

### ❌ Hiệu năng thấp

**Kiểm tra CoreML provider:**
```bash
source venv/bin/activate
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Phải thấy: `['CoreMLExecutionProvider', 'CPUExecutionProvider']`

**Nếu không có CoreML:**
```bash
pip uninstall -y onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.16.3
```

### ❌ Memory pressure cao

```bash
# Giảm max-memory
python run.py --max-memory 32 --execution-provider coreml ...
```

---

## 💡 Tips & Tricks

### 1. Tạo Aliases Nhanh

Thêm vào `~/.zshrc` hoặc `~/.bash_profile`:

```bash
# Deep-Live-Cam aliases
alias dlc-webcam='cd ~/Documents/Deep-Live-Cam && ./start-webcam.sh'
alias dlc-quality='cd ~/Documents/Deep-Live-Cam && ./start-quality.sh'
alias dlc-speed='cd ~/Documents/Deep-Live-Cam && ./start-speed.sh'
alias dlc-balanced='cd ~/Documents/Deep-Live-Cam && ./start-balanced.sh'
```

Sau đó:
```bash
source ~/.zshrc
dlc-webcam  # Chạy ngay!
```

### 2. Quick Process Function

Thêm vào `~/.zshrc`:

```bash
dlc-process() {
    cd ~/Documents/Deep-Live-Cam
    source venv/bin/activate
    python run.py \
      --source "$1" \
      --target "$2" \
      --output "$3" \
      --execution-provider coreml \
      --max-memory 48 \
      --execution-threads 10 \
      --frame-processor face_swapper \
      --keep-fps \
      --keep-audio
}
```

Dùng:
```bash
dlc-process face.jpg input.mp4 output.mp4
```

### 3. Keyboard Shortcuts với Automator

1. Mở Automator
2. New Document → Quick Action
3. Thêm "Run Shell Script"
4. Paste:
   ```bash
   cd ~/Documents/Deep-Live-Cam
   ./start-webcam.sh
   ```
5. Save as "Launch Deep-Live-Cam"
6. System Settings → Keyboard → Shortcuts → Assign key

### 4. Tối Ưu Storage

Models và temp files có thể chiếm nhiều dung lượng:

```bash
# Xóa temporary frames sau khi xong
# Không dùng flag --keep-frames

# Kiểm tra dung lượng
du -sh models/
du -sh output/
```

### 5. Quality vs File Size

| Quality | File Size (10 min 1080p) | Visual Difference |
|---------|--------------------------|-------------------|
| 4 | ~800 MB | Xuất sắc |
| 8 | ~500 MB | Rất tốt |
| 12 | ~350 MB | Tốt ⭐ |
| 18 | ~200 MB | OK |
| 20 | ~150 MB | Chấp nhận được |

**Khuyến nghị:** Quality 12 cho balanced mode

### 6. Best Source Images

Để có kết quả tốt nhất:
- ✅ Ảnh chính diện, ánh sáng tốt
- ✅ Độ phân giải cao (>1024x1024)
- ✅ Không đeo kính, không bị che mặt
- ✅ Expression trung tính
- ❌ Tránh ảnh góc nghiêng
- ❌ Tránh ảnh mờ, tối

---

## 🎓 Học Thêm

### Command Line Arguments Đầy Đủ

```bash
python run.py --help
```

**Một số options hữu ích:**

```bash
# NSFW filter
--nsfw-filter

# Video encoder options
--video-encoder libx264    # H.264 (fast, compatible)
--video-encoder libx265    # H.265 (better quality, smaller size)
--video-encoder libvpx-vp9 # VP9 (for web)

# Video quality (0-51, lower = better)
--video-quality 4   # Rất cao
--video-quality 12  # Cân bằng
--video-quality 20  # Nhanh

# Memory limit
--max-memory 48     # 48 GB

# Execution threads
--execution-threads 10

# Frame processors
--frame-processor face_swapper
--frame-processor face_swapper face_enhancer
```

### Script Template Tùy Chỉnh

```bash
#!/bin/bash
# my-custom-script.sh

cd ~/Documents/Deep-Live-Cam
source venv/bin/activate

python run.py \
  --source "$1" \
  --target "$2" \
  --output "$3" \
  --execution-provider coreml \
  --max-memory 48 \
  --execution-threads 10 \
  --frame-processor face_swapper \
  --many-faces \
  --keep-fps \
  --keep-audio \
  --video-encoder libx264 \
  --video-quality 12
```

---

## 📞 Hỗ Trợ

- **GitHub Issues:** https://github.com/hacksider/Deep-Live-Cam/issues
- **Documentation:** https://github.com/hacksider/Deep-Live-Cam
- **Discord:** Check repository for link

---

## ⚠️ Disclaimer

- Chỉ sử dụng với khuôn mặt đã được đồng ý
- Đánh dấu rõ ràng output là deepfake khi chia sẻ
- Tuân thủ luật pháp địa phương
- Sử dụng có trách nhiệm

---

## 🎉 Chúc Mừng!

Bạn đã sẵn sàng sử dụng Deep-Live-Cam với hiệu năng tối đa trên MacBook M1 Pro Max!

**Khuyến nghị để bắt đầu:**
1. Chạy `./start-balanced.sh` cho lần đầu
2. Test với video ngắn (~30s)
3. Thử các chế độ khác để tìm balance phù hợp

Happy face swapping! 🎭✨
