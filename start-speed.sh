#!/bin/bash

# ============================================================
# Deep-Live-Cam - High Speed Video Processing Mode
# Optimized for M1 Pro Max (64GB RAM, 32 GPU Cores)
# Tốc độ cao nhất - xử lý nhanh, chất lượng tốt
# ============================================================

echo "⚡ Starting Deep-Live-Cam - SPEED Mode"
echo "Chế độ tốc độ cao nhất cho M1 Pro Max"
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

echo "⚡ Cấu hình SPEED:"
echo "   - Execution Provider: CoreML (Apple Neural Engine)"
echo "   - Max Memory: 40 GB"
echo "   - Execution Threads: 12"
echo "   - Frame Processors: Face Swapper only"
echo "   - Video Encoder: H.264 (Fast)"
echo "   - Video Quality: 20 (Fast encode)"
echo "   - Many Faces: Enabled"
echo ""
echo "📝 Cách sử dụng:"
echo "   ./start-speed.sh"
echo ""
echo "   Hoặc với tham số:"
echo "   ./start-speed.sh [source.jpg] [target.mp4] [output.mp4]"
echo ""
echo "💡 Lưu ý:"
echo "   - Chế độ này ưu tiên tốc độ xử lý"
echo "   - Không dùng Face Enhancer để tăng tốc"
echo "   - H.264 với encoding nhanh"
echo "   - Có thể đạt 30-45 FPS với video 1080p"
echo ""

# Nếu có tham số dòng lệnh
if [ $# -eq 3 ]; then
    SOURCE_PATH="$1"
    TARGET_PATH="$2"
    OUTPUT_PATH="$3"

    if [ ! -f "$SOURCE_PATH" ]; then
        echo "❌ File source không tồn tại: $SOURCE_PATH"
        exit 1
    fi

    if [ ! -f "$TARGET_PATH" ]; then
        echo "❌ File target không tồn tại: $TARGET_PATH"
        exit 1
    fi

    echo "📂 Input:"
    echo "   Source: $SOURCE_PATH"
    echo "   Target: $TARGET_PATH"
    echo "   Output: $OUTPUT_PATH"
    echo ""
    echo "⏳ Đang xử lý với tốc độ cao..."
    echo ""

    python run.py \
      --source "$SOURCE_PATH" \
      --target "$TARGET_PATH" \
      --output "$OUTPUT_PATH" \
      --execution-provider coreml \
      --max-memory 40 \
      --execution-threads 12 \
      --frame-processor face_swapper \
      --many-faces \
      --keep-fps \
      --keep-audio \
      --video-encoder libx264 \
      --video-quality 20

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Xử lý hoàn tất với tốc độ cao!"
        echo "📹 Output: $OUTPUT_PATH"
    else
        echo ""
        echo "❌ Có lỗi xảy ra trong quá trình xử lý"
    fi
else
    echo "⏳ Đang khởi động GUI..."
    echo ""

    # Chạy GUI với cấu hình speed
    python run.py \
      --execution-provider coreml \
      --max-memory 40 \
      --execution-threads 12 \
      --frame-processor face_swapper
fi

echo ""
echo "👋 Deep-Live-Cam đã tắt"
