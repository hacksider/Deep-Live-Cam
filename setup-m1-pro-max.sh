#!/bin/bash

# ============================================================
# Deep-Live-Cam Setup Script for MacBook M1 Pro Max
# Optimized for 64GB RAM and 32 GPU Cores
# ============================================================

echo "=================================================="
echo "Deep-Live-Cam Setup for M1 Pro Max"
echo "=================================================="
echo ""

# Kiểm tra xem đang chạy trên macOS không
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ Script này chỉ dành cho macOS!"
    exit 1
fi

# Kiểm tra xem đang chạy trên Apple Silicon không
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "❌ Script này chỉ dành cho Apple Silicon (M1/M2/M3)!"
    exit 1
fi

echo "✅ Phát hiện Apple Silicon Mac"
echo ""

# Kiểm tra Homebrew
echo "📦 Kiểm tra Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew chưa được cài đặt. Đang cài đặt..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew đã được cài đặt"
fi
echo ""

# Cài đặt Python 3.10
echo "🐍 Kiểm tra Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    echo "⚠️  Python 3.10 chưa được cài đặt. Đang cài đặt..."
    brew install python@3.10
else
    echo "✅ Python 3.10 đã được cài đặt"
fi
echo ""

# Cài đặt ffmpeg
echo "🎬 Kiểm tra ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg chưa được cài đặt. Đang cài đặt..."
    brew install ffmpeg
else
    echo "✅ ffmpeg đã được cài đặt"
fi
echo ""

# Cài đặt python-tk
echo "🖼️  Kiểm tra python-tk..."
brew install python-tk@3.10 2>/dev/null || echo "✅ python-tk đã được cài đặt"
echo ""

# Tạo virtual environment
echo "🔧 Tạo virtual environment..."
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
    echo "✅ Virtual environment đã được tạo"
else
    echo "✅ Virtual environment đã tồn tại"
fi
echo ""

# Kích hoạt virtual environment
echo "⚡ Kích hoạt virtual environment..."
source venv/bin/activate

# Nâng cấp pip
echo "📦 Nâng cấp pip..."
pip install --upgrade pip
echo ""

# Cài đặt dependencies
echo "📚 Cài đặt dependencies cho macOS..."
if [ -f "requirements-macos.txt" ]; then
    echo "   Sử dụng requirements-macos.txt (tối ưu cho Apple Silicon)"
    pip install -r requirements-macos.txt
else
    echo "   Sử dụng requirements.txt"
    pip install -r requirements.txt
fi
echo ""

# Tối ưu hóa cho Apple Silicon - cài đặt onnxruntime-silicon
echo "🚀 Tối ưu hóa cho Apple Silicon..."
pip uninstall -y onnxruntime onnxruntime-silicon onnxruntime-coreml 2>/dev/null
pip install onnxruntime-silicon==1.16.3
echo "✅ Đã cài đặt onnxruntime-silicon"
echo ""

# Tạo thư mục models nếu chưa có
echo "📁 Kiểm tra thư mục models..."
mkdir -p models
echo ""

# Kiểm tra models
echo "🔍 Kiểm tra models..."
if [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "⚠️  GFPGANv1.4.pth chưa được tải xuống"
    echo "   Đang tải xuống... (có thể mất vài phút)"
    curl -L -o models/GFPGANv1.4.pth "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"
    echo "✅ Đã tải GFPGANv1.4.pth"
else
    echo "✅ GFPGANv1.4.pth đã tồn tại"
fi

if [ ! -f "models/inswapper_128.onnx" ]; then
    echo "⚠️  inswapper_128.onnx chưa được tải xuống"
    echo "   Đang tải xuống... (có thể mất vài phút)"
    curl -L -o models/inswapper_128.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
    echo "✅ Đã tải inswapper_128.onnx"
else
    echo "✅ inswapper_128.onnx đã tồn tại"
fi
echo ""

# Tạo các script chạy nhanh
echo "📝 Tạo các script chạy nhanh..."
chmod +x start-*.sh 2>/dev/null
echo "✅ Đã cấp quyền thực thi cho các script"
echo ""

# Kiểm tra cài đặt
echo "🧪 Kiểm tra cài đặt..."
python -c "import onnxruntime; print('Available providers:', onnxruntime.get_available_providers())" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Cài đặt thành công!"
else
    echo "⚠️  Có lỗi xảy ra trong quá trình kiểm tra"
fi
echo ""

echo "=================================================="
echo "✨ Cài đặt hoàn tất!"
echo "=================================================="
echo ""
echo "📖 Các script có sẵn:"
echo "   ./start-webcam.sh          - Chế độ webcam realtime"
echo "   ./start-quality.sh         - Xử lý video chất lượng cao nhất"
echo "   ./start-speed.sh           - Xử lý video nhanh nhất"
echo "   ./start-balanced.sh        - Cân bằng tốc độ và chất lượng"
echo ""
echo "📝 Để chạy thủ công:"
echo "   source venv/bin/activate"
echo "   python run.py --execution-provider coreml --max-memory 48"
echo ""
echo "🎉 Chúc bạn sử dụng vui vẻ!"
