#!/bin/bash

# ============================================================
# Deep-Live-Cam - High Quality Video Processing Mode
# Optimized for M1 Pro Max (64GB RAM, 32 GPU Cores)
# Chất lượng cao nhất - xử lý chậm hơn nhưng kết quả tốt nhất
# ============================================================

echo "💎 Starting Deep-Live-Cam - QUALITY Mode"
echo "Chế độ chất lượng cao nhất cho M1 Pro Max"
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

echo "⚡ Cấu hình QUALITY:"
echo "   - Execution Provider: CoreML (Apple Neural Engine)"
echo "   - Max Memory: 56 GB"
echo "   - Execution Threads: 8"
echo "   - Frame Processors: Face Swapper + Face Enhancer"
echo "   - Video Encoder: H.265 (HEVC)"
echo "   - Video Quality: 4 (Rất cao)"
echo "   - Many Faces: Enabled"
echo ""
echo "📝 Cách sử dụng:"
echo "   ./start-quality.sh"
echo ""
echo "   Hoặc với tham số:"
echo "   ./start-quality.sh [source.jpg] [target.mp4] [output.mp4]"
echo ""
echo "💡 Lưu ý:"
echo "   - Chế độ này ưu tiên chất lượng, tốc độ xử lý chậm hơn"
echo "   - Face Enhancer sẽ tăng độ chi tiết khuôn mặt"
echo "   - H.265 cho file nhỏ hơn với chất lượng tốt hơn"
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
    echo "⏳ Đang xử lý..."
    echo ""

    python run.py \
      --source "$SOURCE_PATH" \
      --target "$TARGET_PATH" \
      --output "$OUTPUT_PATH" \
      --execution-provider coreml \
      --max-memory 56 \
      --execution-threads 8 \
      --frame-processor face_swapper face_enhancer \
      --many-faces \
      --keep-fps \
      --keep-audio \
      --video-encoder libx265 \
      --video-quality 4

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Xử lý hoàn tất!"
        echo "📹 Output: $OUTPUT_PATH"
    else
        echo ""
        echo "❌ Có lỗi xảy ra trong quá trình xử lý"
    fi
else
    echo "⏳ Đang khởi động GUI..."
    echo ""

    # Chạy GUI với cấu hình quality
    python run.py \
      --execution-provider coreml \
      --max-memory 56 \
      --execution-threads 8 \
      --frame-processor face_swapper face_enhancer
fi

echo ""
echo "👋 Deep-Live-Cam đã tắt"
