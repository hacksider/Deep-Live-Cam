#!/bin/bash

# ============================================================
# Deep-Live-Cam - Webcam Realtime Mode
# Optimized for M1 Pro Max (64GB RAM, 32 GPU Cores)
# ============================================================

echo "🎥 Starting Deep-Live-Cam - Webcam Mode"
echo "Tối ưu cho M1 Pro Max với hiệu năng cao"
echo ""

# Chuyển đến thư mục script
cd "$(dirname "$0")"

# Kiểm tra virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment không tồn tại!"
    echo "Vui lòng chạy ./setup-m1-pro-max.sh trước"
    exit 1
fi

# Kích hoạt virtual environment
source venv/bin/activate

# Kiểm tra models
if [ ! -f "models/GFPGANv1.4.pth" ] || [ ! -f "models/inswapper_128.onnx" ]; then
    echo "❌ Models chưa được tải xuống!"
    echo "Vui lòng chạy ./setup-m1-pro-max.sh trước"
    exit 1
fi

echo "⚡ Cấu hình:"
echo "   - Execution Provider: CoreML (Apple Neural Engine)"
echo "   - Max Memory: 48 GB"
echo "   - Execution Threads: 10"
echo "   - Many Faces: Enabled"
echo "   - Live Resizable: Enabled"
echo "   - Live Mirror: Enabled"
echo ""
echo "📝 Hướng dẫn:"
echo "   1. Chọn ảnh khuôn mặt nguồn (source face)"
echo "   2. Nhấn nút 'Live' để bắt đầu"
echo "   3. Cho phép truy cập camera khi được hỏi"
echo "   4. Sử dụng OBS để capture và stream"
echo ""
echo "⏳ Đang khởi động..."
echo ""

# Chạy với cấu hình tối ưu cho webcam realtime
python run.py \
  --execution-provider coreml \
  --max-memory 48 \
  --execution-threads 10 \
  --many-faces \
  --live-resizable \
  --live-mirror

echo ""
echo "👋 Deep-Live-Cam đã tắt"
