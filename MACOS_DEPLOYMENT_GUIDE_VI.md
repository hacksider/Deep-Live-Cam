# Hướng Dẫn Triển Khai Deep-Live-Cam trên MacBook

Hướng dẫn chi tiết từng bước để cài đặt và chạy Deep-Live-Cam trên macOS.

## Mục Lục
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Chuẩn Bị](#chuẩn-bị)
- [Cài Đặt Bước 1: Cài Đặt Công Cụ Cần Thiết](#bước-1-cài-đặt-công-cụ-cần-thiết)
- [Bước 2: Clone Repository](#bước-2-clone-repository-từ-github)
- [Bước 3: Tạo Môi Trường Ảo](#bước-3-tạo-môi-trường-ảo)
- [Bước 4: Cài Đặt Dependencies](#bước-4-cài-đặt-dependencies)
- [Bước 5: Tải Models](#bước-5-tải-models)
- [Bước 6: Chạy Ứng Dụng](#bước-6-chạy-ứng-dụng)
- [Tăng Tốc GPU (Tùy Chọn)](#tăng-tốc-gpu-cho-apple-silicon)
- [Xử Lý Sự Cố](#xử-lý-sự-cố)

---

## Yêu Cầu Hệ Thống

### Tối Thiểu
- **macOS**: 10.15 (Catalina) trở lên
- **RAM**: 8GB trở lên (khuyến nghị 16GB)
- **Dung lượng ổ cứng**: 5GB trống
- **Processor**: Intel hoặc Apple Silicon (M1/M2/M3)

### Khuyến Nghị
- **macOS**: 12.0 (Monterey) hoặc mới hơn
- **RAM**: 16GB trở lên
- **Chip**: Apple Silicon (M1/M2/M3) để có hiệu suất tốt nhất với CoreML

---

## Chuẩn Bị

Trước khi bắt đầu, đảm bảo bạn có:
- Kết nối internet ổn định
- Quyền quản trị (admin) trên máy Mac
- Kiến thức cơ bản về Terminal

---

## Bước 1: Cài Đặt Công Cụ Cần Thiết

### 1.1. Cài Đặt Homebrew

Homebrew là trình quản lý gói cho macOS. Mở **Terminal** và chạy lệnh sau:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Sau khi cài đặt xong, làm theo hướng dẫn trên màn hình để thêm Homebrew vào PATH của bạn.

**Kiểm tra cài đặt:**
```bash
brew --version
```

### 1.2. Cài Đặt Python 3.10

Deep-Live-Cam khuyến nghị sử dụng Python 3.10:

```bash
brew install python@3.10
```

**Kiểm tra cài đặt:**
```bash
python3.10 --version
```

Bạn sẽ thấy kết quả tương tự: `Python 3.10.x`

### 1.3. Cài Đặt Git

```bash
brew install git
```

**Kiểm tra cài đặt:**
```bash
git --version
```

### 1.4. Cài Đặt FFmpeg

FFmpeg được sử dụng để xử lý video:

```bash
brew install ffmpeg
```

**Kiểm tra cài đặt:**
```bash
ffmpeg -version
```

### 1.5. Cài Đặt python-tk

Thư viện này cần thiết cho giao diện đồ họa:

```bash
brew install python-tk@3.10
```

---

## Bước 2: Clone Repository từ GitHub

### 2.1. Chọn Thư Mục Làm Việc

Mở Terminal và di chuyển đến thư mục bạn muốn lưu dự án. Ví dụ:

```bash
cd ~/Documents
```

hoặc tạo thư mục mới:

```bash
mkdir ~/Projects
cd ~/Projects
```

### 2.2. Clone Repository

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
```

### 2.3. Di Chuyển Vào Thư Mục Dự Án

```bash
cd Deep-Live-Cam
```

---

## Bước 3: Tạo Môi Trường Ảo

Việc sử dụng môi trường ảo (virtual environment) giúp tránh xung đột với các gói Python khác trên hệ thống.

### 3.1. Tạo Virtual Environment

```bash
python3.10 -m venv venv
```

Lệnh này sẽ tạo một thư mục `venv` chứa môi trường Python độc lập.

### 3.2. Kích Hoạt Virtual Environment

```bash
source venv/bin/activate
```

Sau khi kích hoạt, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh:

```
(venv) username@MacBook Deep-Live-Cam %
```

**Lưu ý**: Bạn cần kích hoạt virtual environment mỗi khi mở Terminal mới để làm việc với dự án.

### 3.3. Nâng Cấp pip

```bash
pip install --upgrade pip
```

---

## Bước 4: Cài Đặt Dependencies

### 4.1. Cài Đặt Requirements

⚠️ **QUAN TRỌNG**: Trên macOS, sử dụng file `requirements-macos.txt` thay vì `requirements.txt`

```bash
pip install -r requirements-macos.txt
```

Quá trình này có thể mất 5-15 phút tùy thuộc vào tốc độ internet và cấu hình máy.

**Lưu ý quan trọng**:
- File `requirements-macos.txt` đã được tối ưu cho macOS (không có CUDA)
- PyTorch sẽ được cài đặt phiên bản phù hợp cho macOS
- Nếu bạn có Apple Silicon (M1/M2/M3), `onnxruntime-silicon` sẽ được cài tự động
- Nếu bạn có MacBook Intel, `onnxruntime` thông thường sẽ được cài

### 4.2. Xác Nhận Cài Đặt Thành Công

Kiểm tra các gói quan trọng:

```bash
pip list | grep -E "torch|onnx|opencv"
```

Bạn sẽ thấy danh sách các gói đã được cài đặt.

---

## Bước 5: Tải Models

Deep-Live-Cam cần 2 model để hoạt động. Bạn cần tải chúng về và đặt vào thư mục `models`.

### 5.1. Tạo Thư Mục Models (nếu chưa có)

```bash
mkdir -p models
```

### 5.2. Tải GFPGANv1.4

**Cách 1: Sử dụng trình duyệt**
1. Mở link: [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. File sẽ tự động tải về thư mục Downloads
3. Di chuyển file vào thư mục models:

```bash
mv ~/Downloads/GFPGANv1.4.pth models/
```

**Cách 2: Sử dụng Terminal**

```bash
cd models
curl -L -o GFPGANv1.4.pth "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"
cd ..
```

### 5.3. Tải inswapper_128_fp16.onnx

**Khuyến nghị**: Sử dụng phiên bản thay thế từ facefusion (ổn định hơn):

```bash
cd models
curl -L -o inswapper_128.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
cd ..
```

### 5.4. Xác Nhận Đã Tải Đầy Đủ

```bash
ls -lh models/
```

Bạn sẽ thấy 2 file:
- `GFPGANv1.4.pth` (~332 MB)
- `inswapper_128.onnx` (~256 MB)

---

## Bước 6: Chạy Ứng Dụng

### 6.1. Chạy Lần Đầu

Đảm bảo bạn đang ở trong thư mục `Deep-Live-Cam` và virtual environment đã được kích hoạt (`(venv)` xuất hiện).

```bash
python run.py
```

**Lần chạy đầu tiên**: Ứng dụng sẽ tải xuống thêm một số models (~300MB), quá trình này có thể mất 5-10 phút.

### 6.2. Giao Diện Ứng Dụng

Sau khi khởi động thành công, giao diện đồ họa sẽ xuất hiện với các tùy chọn:
- **Source**: Chọn ảnh khuôn mặt nguồn
- **Target**: Chọn ảnh/video đích
- **Start**: Bắt đầu xử lý
- **Live**: Chế độ webcam thời gian thực

### 6.3. Chế Độ Sử Dụng

#### **Chế độ Ảnh/Video:**
1. Click "Select a face" để chọn ảnh khuôn mặt nguồn
2. Click "Select a target" để chọn ảnh hoặc video đích
3. Click "Start" để bắt đầu xử lý
4. Kết quả sẽ được lưu trong thư mục có tên giống file đích

#### **Chế độ Webcam:**
1. Click "Select a face" để chọn ảnh khuôn mặt nguồn
2. Click "Live" để bắt đầu
3. Đợi 10-30 giây để preview xuất hiện
4. Sử dụng công cụ quay màn hình như OBS để stream

---

## Tăng Tốc GPU cho Apple Silicon

Nếu bạn có MacBook với chip Apple Silicon (M1/M2/M3), bạn có thể tăng hiệu suất đáng kể bằng cách sử dụng CoreML.

### Cài Đặt CoreML Support

1. **Gỡ cài đặt onnxruntime mặc định:**

```bash
pip uninstall onnxruntime onnxruntime-silicon
```

2. **Cài đặt onnxruntime-silicon:**

```bash
pip install onnxruntime-silicon==1.16.3
```

### Chạy với CoreML

```bash
python run.py --execution-provider coreml
```

**Lưu ý**:
- CoreML chỉ hoạt động trên Apple Silicon (M1/M2/M3)
- Nếu bạn có MacBook Intel, hãy bỏ qua phần này và chạy với CPU thông thường

---

## Xử Lý Sự Cố

### Lỗi 1: "command not found: python"

**Giải pháp**: Sử dụng `python3` hoặc `python3.10` thay vì `python`

```bash
python3.10 run.py
```

### Lỗi 2: "No module named 'tkinter'"

**Giải pháp**: Cài đặt python-tk

```bash
brew install python-tk@3.10
```

Sau đó kích hoạt lại virtual environment:

```bash
deactivate
source venv/bin/activate
```

### Lỗi 3: "Could not find a version that satisfies the requirement torch==2.0.1+cu118"

**Triệu chứng**:
```
ERROR: Could not find a version that satisfies the requirement torch==2.0.1+cu118
ERROR: No matching distribution found for torch==2.0.1+cu118
```

**Nguyên nhân**: Bạn đang sử dụng file `requirements.txt` thay vì `requirements-macos.txt`. File `requirements.txt` chứa phiên bản PyTorch với CUDA (dành cho Nvidia GPU), không tương thích với macOS.

**Giải pháp**:

1. **Xóa virtual environment hiện tại**:
```bash
deactivate
rm -rf venv
```

2. **Tạo lại virtual environment**:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

3. **Cài đặt với file đúng cho macOS**:
```bash
pip install -r requirements-macos.txt
```

### Lỗi 4: "ModuleNotFoundError: No module named 'cv2'"

**Giải pháp**: Cài đặt lại opencv-python

```bash
pip install --upgrade opencv-python==4.8.1.78
```

### Lỗi 5: Virtual environment không kích hoạt

**Triệu chứng**: Không thấy `(venv)` ở đầu dòng lệnh

**Giải pháp**:

```bash
cd ~/Projects/Deep-Live-Cam  # Thay đổi đường dẫn nếu cần
source venv/bin/activate
```

### Lỗi 6: "Permission denied"

**Giải pháp**: Cấp quyền cho Terminal truy cập Camera và Microphone

1. Vào **System Preferences** (Tùy chọn Hệ thống)
2. Chọn **Security & Privacy** (Bảo mật & Quyền riêng tư)
3. Tab **Privacy** (Quyền riêng tư)
4. Chọn **Camera** và cho phép Terminal
5. Chọn **Microphone** và cho phép Terminal (nếu cần)

### Lỗi 7: Hiệu suất chậm

**Giải pháp**:

1. **Nếu có Apple Silicon**, hãy sử dụng CoreML (xem phần [Tăng Tốc GPU](#tăng-tốc-gpu-cho-apple-silicon))

2. **Giảm độ phân giải**: Sử dụng tham số `--live-resizable`

```bash
python run.py --live-resizable
```

3. **Giới hạn RAM**: Nếu máy có RAM thấp

```bash
python run.py --max-memory 4
```

### Lỗi 8: Models không tải được

**Giải pháp**: Kiểm tra lại đường dẫn và kích thước file

```bash
ls -lh models/
```

Nếu file bị thiếu hoặc kích thước không đúng, tải lại models theo [Bước 5](#bước-5-tải-models).

### Lỗi 9: "RuntimeError: No ffmpeg exe could be found"

**Giải pháp**: Cài đặt lại ffmpeg

```bash
brew reinstall ffmpeg
```

Sau đó kiểm tra:

```bash
which ffmpeg
```

### Lỗi 10: "OpenCV: camera failed to properly initialize!"

**Triệu chứng**:
```
OpenCV: out device of bound (0-0): 1
OpenCV: camera failed to properly initialize!
OpenCV: out device of bound (0-0): 2
OpenCV: camera failed to properly initialize!
```

**Nguyên nhân**:
- Ứng dụng không có quyền truy cập camera
- Camera đang được sử dụng bởi ứng dụng khác
- Vấn đề với cv2_enumerate_cameras trên macOS

**Giải pháp**:

**1. Kiểm tra và cấp quyền camera cho Terminal/Python:**

a. Vào **System Settings** (hoặc System Preferences trên macOS cũ):
   - Chọn **Privacy & Security**
   - Chọn **Camera** (ở sidebar bên trái)
   - Đảm bảo **Terminal** được bật
   - Nếu dùng VS Code hoặc IDE khác, cũng cần bật cho ứng dụng đó

b. Khởi động lại Terminal sau khi cấp quyền

**2. Đóng các ứng dụng đang dùng camera:**

```bash
# Kiểm tra ứng dụng nào đang dùng camera
lsof | grep "Camera"
```

Đóng các ứng dụng như:
- Zoom, Skype, Microsoft Teams
- Photo Booth, FaceTime
- Bất kỳ ứng dụng video call nào khác

**3. Thử chạy ở chế độ ảnh/video trước (không dùng camera):**

```bash
# Chạy với file ảnh/video thay vì camera
python run.py
# Sau đó chọn ảnh nguồn và ảnh/video đích, click "Start"
```

**4. Test camera với script đơn giản:**

Tạo file test `test_camera.py`:

```python
import cv2

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✓ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("✓ Frame captured successfully!")
        print(f"Frame shape: {frame.shape}")
    else:
        print("✗ Failed to capture frame")
else:
    print("✗ Failed to open camera")

cap.release()
```

Chạy test:

```bash
python test_camera.py
```

**5. Nếu vẫn không hoạt động, thử gỡ và cài lại opencv:**

```bash
pip uninstall opencv-python cv2_enumerate_cameras -y
pip install opencv-python==4.8.1.78
pip install cv2_enumerate_cameras==1.1.15
```

**6. Thử với camera index khác:**

Một số máy Mac có nhiều camera (internal + external). Thử các index khác:

```bash
# Test với các camera index khác
# Sửa file run.py tạm thời hoặc thử script test với index 1, 2
```

**7. Workaround - Chạy chế độ không cần camera:**

Nếu bạn chỉ cần xử lý ảnh/video (không cần webcam live):

```bash
# Chạy với tham số source và target từ command line
python run.py -s path/to/source.jpg -t path/to/target.mp4 -o output.mp4
```

**Lưu ý cho macOS Ventura (13.0) trở lên:**
- Apple đã tăng cường bảo mật camera
- Bạn có thể cần cho phép "Screen Recording" permission nếu dùng camera trong một số trường hợp
- Vào **System Settings > Privacy & Security > Screen Recording** và bật cho Terminal

---

## Các Tham Số Command Line Hữu Ích

```bash
# Chạy với nhiều khuôn mặt
python run.py --many-faces

# Giữ nguyên FPS gốc
python run.py --keep-fps

# Giữ nguyên audio gốc
python run.py --keep-audio

# Lọc nội dung NSFW
python run.py --nsfw-filter

# Giới hạn RAM (đơn vị: GB)
python run.py --max-memory 8

# Chạy với CoreML (Apple Silicon)
python run.py --execution-provider coreml

# Kết hợp nhiều tham số
python run.py --execution-provider coreml --many-faces --keep-fps
```

---

## Gỡ Cài Đặt

Nếu bạn muốn gỡ bỏ Deep-Live-Cam:

```bash
# Thoát virtual environment (nếu đang trong)
deactivate

# Xóa thư mục dự án
cd ~/Projects  # Thay đổi đường dẫn nếu cần
rm -rf Deep-Live-Cam

# (Tùy chọn) Gỡ các công cụ đã cài
brew uninstall python@3.10 ffmpeg
```

---

## Cập Nhật Dự Án

Để cập nhật lên phiên bản mới nhất:

```bash
cd ~/Projects/Deep-Live-Cam  # Thay đổi đường dẫn nếu cần
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## Tài Nguyên Bổ Sung

- **Repository chính**: https://github.com/hacksider/Deep-Live-Cam
- **Issues & Hỗ trợ**: https://github.com/hacksider/Deep-Live-Cam/issues
- **Discord**: Tham gia Discord để được hỗ trợ từ cộng đồng

---

## Lưu Ý Quan Trọng

⚠️ **Disclaimer**:
- Phần mềm này chỉ nên được sử dụng cho mục đích hợp pháp và có đạo đức
- Luôn xin phép trước khi sử dụng khuôn mặt của người khác
- Đánh dấu rõ ràng nội dung deepfake khi chia sẻ công khai
- Người phát triển không chịu trách nhiệm về việc sử dụng sai mục đích

---

## Kết Luận

Bạn đã hoàn thành việc cài đặt Deep-Live-Cam trên MacBook! Giờ đây bạn có thể:
- Thay đổi khuôn mặt trong ảnh và video
- Sử dụng chế độ webcam real-time
- Tối ưu hiệu suất với CoreML (nếu có Apple Silicon)

Nếu gặp bất kỳ vấn đề nào, hãy tham khảo phần [Xử Lý Sự Cố](#xử-lý-sự-cố) hoặc tạo issue trên GitHub.

**Chúc bạn sử dụng thành công!** 🎉
