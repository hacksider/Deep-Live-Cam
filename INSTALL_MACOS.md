# 🍎 Hướng Dẫn Cài Đặt Cho macOS (Apple Silicon)

## ⚠️ Nếu Bạn Gặp Lỗi Với requirements.txt

Nếu bạn gặp lỗi:
```
ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu128
```

**Nguyên nhân:** File `requirements.txt` gốc chứa dependencies cho CUDA (Windows/Linux), không tương thích với macOS.

**Giải pháp:** Sử dụng `requirements-macos.txt` thay thế.

---

## 🚀 Cài Đặt Nhanh (Khuyến Nghị)

### Phương Pháp 1: Dùng Script Tự Động (Dễ Nhất)

```bash
chmod +x setup-m1-pro-max.sh
./setup-m1-pro-max.sh
```

Script này sẽ tự động:
- ✅ Cài đặt tất cả dependencies từ `requirements-macos.txt`
- ✅ Tối ưu hóa cho Apple Silicon
- ✅ Tải models tự động
- ✅ Cấu hình CoreML

---

### Phương Pháp 2: Cài Đặt Thủ Công

#### Bước 1: Cài Đặt Homebrew và Dependencies

```bash
# Cài Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Cài Python 3.10, ffmpeg, python-tk
brew install python@3.10 ffmpeg python-tk@3.10
```

#### Bước 2: Tạo Virtual Environment

```bash
# Tạo venv
python3.10 -m venv venv

# Kích hoạt
source venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip
```

#### Bước 3: Cài Đặt Dependencies Cho macOS

```bash
# QUAN TRỌNG: Dùng requirements-macos.txt thay vì requirements.txt
pip install -r requirements-macos.txt
```

#### Bước 4: Cài Đặt ONNX Runtime cho Apple Silicon

```bash
# Gỡ các phiên bản cũ
pip uninstall -y onnxruntime onnxruntime-silicon onnxruntime-coreml

# Cài phiên bản tối ưu cho M1/M2/M3
pip install onnxruntime-silicon==1.16.3
```

#### Bước 5: Tải Models

```bash
# Tạo thư mục
mkdir -p models

# Tải GFPGANv1.4
curl -L -o models/GFPGANv1.4.pth \
  "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"

# Tải inswapper_128
curl -L -o models/inswapper_128.onnx \
  "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
```

#### Bước 6: Kiểm Tra Cài Đặt

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Kết quả mong đợi:
```
['CoreMLExecutionProvider', 'CPUExecutionProvider']
```

✅ Nếu thấy `CoreMLExecutionProvider` → Thành công!

---

## 🎯 Chạy Ứng Dụng

### Cách 1: Dùng Scripts Có Sẵn

```bash
# Webcam mode
./start-webcam.sh

# Balanced mode (khuyến nghị)
./start-balanced.sh

# Quality mode
./start-quality.sh

# Speed mode
./start-speed.sh
```

### Cách 2: Command Line

```bash
source venv/bin/activate

python run.py \
  --execution-provider coreml \
  --max-memory 48 \
  --execution-threads 10
```

---

## 🔧 Xử Lý Lỗi Thường Gặp

### Lỗi 1: "torch==2.0.1+cu118" không tìm thấy

**Nguyên nhân:** Đang dùng `requirements.txt` thay vì `requirements-macos.txt`

**Giải pháp:**
```bash
pip install -r requirements-macos.txt
```

### Lỗi 2: "tkinter module not found"

**Giải pháp:**
```bash
brew reinstall python-tk@3.10
```

### Lỗi 3: "Could not load CoreML model"

**Giải pháp:**
```bash
pip uninstall -y onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.16.3
```

### Lỗi 4: "numpy" hoặc "opencv" version conflict

**Giải pháp:**
```bash
pip uninstall numpy opencv-python
pip install -r requirements-macos.txt --force-reinstall
```

### Lỗi 5: Permission denied khi truy cập camera

**Giải pháp:**
1. System Settings → Privacy & Security → Camera
2. Bật quyền cho Terminal
3. Restart Terminal

---

## 📊 So Sánh requirements.txt vs requirements-macos.txt

| File | Dành Cho | PyTorch | ONNX Runtime | TensorFlow |
|------|----------|---------|--------------|------------|
| `requirements.txt` | Windows/Linux | 2.0.1+cu118 | onnxruntime-gpu | ✅ |
| `requirements-macos.txt` | macOS | 2.0.1 (CPU) | onnxruntime-silicon | ❌ |

**Lý do loại bỏ TensorFlow trên macOS:**
- Deep-Live-Cam chủ yếu dùng PyTorch và ONNX
- TensorFlow 2.12.1 có vấn đề tương thích với Apple Silicon
- Không cần thiết cho chức năng chính

---

## 💡 Tips

### Nếu Đã Cài Đặt Sai

```bash
# Xóa venv cũ
rm -rf venv

# Tạo lại từ đầu
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-macos.txt
pip uninstall -y onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.16.3
```

### Kiểm Tra Version Python

```bash
python --version
# Phải là: Python 3.10.x
```

### Kiểm Tra Architecture

```bash
uname -m
# Phải là: arm64 (cho Apple Silicon)
```

### Monitor Cài Đặt

```bash
# Xem pip đang cài gì
pip list | grep -E "torch|onnx|opencv"
```

---

## 🎉 Hoàn Tất!

Sau khi cài đặt thành công, bạn có thể:

1. **Chạy webcam mode:**
   ```bash
   ./start-webcam.sh
   ```

2. **Xử lý video:**
   ```bash
   ./start-balanced.sh my_face.jpg input.mp4 output.mp4
   ```

3. **Đọc hướng dẫn chi tiết:**
   ```bash
   cat README_VI_M1.md
   ```

---

## 📞 Hỗ Trợ

Nếu vẫn gặp vấn đề:

1. Đảm bảo dùng `requirements-macos.txt`
2. Kiểm tra Python version (phải 3.10.x)
3. Kiểm tra architecture (phải arm64)
4. Xem logs lỗi chi tiết
5. Thử xóa venv và cài lại từ đầu

**Good luck!** 🚀
