#!/usr/bin/env python3
"""
Performance benchmark script for Deep-Live-Cam
Tests FPS improvements for live mode on macOS
"""

import time
import numpy as np
import cv2
import platform
from collections import deque

# Mock modules.globals for testing
class MockGlobals:
    execution_providers = ["CPUExecutionProvider"]
    live_mode = True
    many_faces = False
    map_faces = False
    simple_map = None
    opacity = 1.0
    face_analyser_engine = "insightface"
    
    fp_ui = {
        "face_enhancer": False,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False
    }

modules = type('Modules', (), {'globals': MockGlobals()})()

def test_frame_processing_speed():
    """Test basic frame processing speed."""
    print("=" * 60)
    print("DEEP-LIVE-CAM PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    print("=" * 60)
    
    # Test 1: Color conversion speed
    print("\n[Test 1] Color Conversion (BGR->RGB)")
    test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    elapsed = time.time() - start
    fps = iterations / elapsed
    print(f"  Average: {1000*elapsed/iterations:.2f}ms per frame")
    print(f"  FPS: {fps:.1f}")
    
    # Test 2: Resize speed
    print("\n[Test 2] Frame Resize (640x360 -> 320x180)")
    start = time.time()
    for _ in range(iterations):
        _ = cv2.resize(test_frame, (320, 180), interpolation=cv2.INTER_AREA)
    elapsed = time.time() - start
    fps = iterations / elapsed
    print(f"  Average: {1000*elapsed/iterations:.2f}ms per frame")
    print(f"  FPS: {fps:.1f}")
    
    # Test 3: Gaussian blur speed
    print("\n[Test 3] Gaussian Blur (kernel=5x5)")
    start = time.time()
    for _ in range(iterations):
        _ = cv2.GaussianBlur(test_frame, (5, 5), 0)
    elapsed = time.time() - start
    fps = iterations / elapsed
    print(f"  Average: {1000*elapsed/iterations:.2f}ms per frame")
    print(f"  FPS: {fps:.1f}")
    
    # Test 4: AddWeighted speed
    print("\n[Test 4] AddWeighted (alpha=0.5)")
    test_frame2 = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    start = time.time()
    for _ in range(iterations):
        _ = cv2.addWeighted(test_frame, 0.5, test_frame2, 0.5, 0)
    elapsed = time.time() - start
    fps = iterations / elapsed
    print(f"  Average: {1000*elapsed/iterations:.2f}ms per frame")
    print(f"  FPS: {fps:.1f}")
    
    # Test 5: Combined pipeline simulation
    print("\n[Test 5] Simulated Live Pipeline (per-frame)")
    processing_times = deque(maxlen=30)
    
    for i in range(60):  # 60 frames test
        frame_start = time.time()
        
        # Simulate processing chain
        frame = test_frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if i % 3 == 0:  # Simulate detection every 3 frames
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.addWeighted(frame, 0.8, test_frame2, 0.2, 0)
        
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        if i >= 10:  # Skip warmup frames
            avg_time = sum(processing_times) / len(processing_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    avg_time = sum(processing_times) / len(processing_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"  Average: {1000*avg_time:.2f}ms per frame")
    print(f"  Estimated FPS: {fps:.1f}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nOptimization Tips:")
    print("1. Use CoreML execution provider on Apple Silicon")
    print("2. Reduce detector size for faster inference")
    print("3. Enable frame skipping if FPS drops below target")
    print("4. Disable enhancers for maximum FPS")
    print("5. Use lower resolution for live mode")
    print("=" * 60)

if __name__ == "__main__":
    test_frame_processing_speed()
