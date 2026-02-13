#!/bin/bash

# ============================================================
# Deep-Live-Cam - Balanced Mode
# Optimized for M1 Pro Max (64GB RAM, 32 GPU Cores)
# Cân bằng giữa tốc độ và chất lượng
# ============================================================

echo "⚖️  Starting Deep-Live-Cam - BALANCED Mode"
echo "Chế độ cân bằng tốc độ và chất lượng cho M1 Pro Max"
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

echo "⚡ Cấu hình BALANCED:"
echo "   - Execution Provider: CoreML (Apple Neural Engine)"
echo "   - Max Memory: 48 GB"
echo "   - Execution Threads: 10"
echo "   - Frame Processors: Face Swapper only"
echo "   - Video Encoder: H.264"
echo "   - Video Quality: 12 (Cân bằng)"
echo "   - Many Faces: Enabled"
echo ""
echo "📝 Cách sử dụng:"
echo "   ./start-balanced.sh"
echo ""
echo "   Hoặc với tham số:"
echo "   ./start-balanced.sh [source.jpg] [target.mp4] [output.mp4]"
echo ""
echo "💡 Lưu ý:"
echo "   - Chế độ khuyến nghị cho sử dụng hàng ngày"
echo "   - Cân bằng tốt giữa tốc độ và chất lượng"
echo "   - Tốc độ ~20-30 FPS với video 1080p"
echo "   - Chất lượng tốt, file size hợp lý"
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
    echo "⏳ Đang xử lý với chế độ cân bằng..."
    echo ""

    python run.py \
      --source "$SOURCE_PATH" \
      --target "$TARGET_PATH" \
      --output "$OUTPUT_PATH" \
      --execution-provider coreml \
      --max-memory 48 \
      --execution-threads 10 \
      --frame-processor face_swapper \
      --many-faces \
      --keep-fps \
      --keep-audio \
      --video-encoder libx264 \
      --video-quality 12

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

    # Chạy GUI với cấu hình balanced
    python run.py \
      --execution-provider coreml \
      --max-memory 48 \
      --execution-threads 10 \
      --frame-processor face_swapper \
      --many-faces
fi

echo ""
echo "👋 Deep-Live-Cam đã tắt"
