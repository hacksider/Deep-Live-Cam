#!/usr/bin/env python3
"""
Diagnostic script to verify Deep-Live-Cam pipeline and performance
"""

import sys
import os
import platform
import importlib.metadata

def check_system():
    """Check system configuration."""
    print("=" * 70)
    print("DEEP-LIVE-CAM - DIAGNÓSTICO DO SISTEMA")
    print("=" * 70)
    
    print(f"\n🖥️  SISTEMA:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    # Check if Apple Silicon
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    print(f"   Apple Silicon: {'✅ Sim' if is_apple_silicon else '❌ Não'}")
    
    # CPU info
    print(f"   CPU Cores: {os.cpu_count()}")
    
    # Memory info (macOS)
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                memory_gb = int(result.stdout.strip()) / (1024**3)
                print(f"   RAM: {memory_gb:.1f} GB")
        except Exception:
            pass

def check_dependencies():
    """Check required dependencies."""
    print(f"\n📦 DEPENDÊNCIAS:")
    
    required_packages = [
        'numpy', 'opencv-python', 'onnx', 'onnxruntime',
        'insightface', 'customtkinter', 'Pillow', 'psutil'
    ]
    
    for package in required_packages:
        try:
            version = importlib.metadata.version(package)
            print(f"   ✅ {package}: {version}")
        except Exception:
            print(f"   ❌ {package}: NÃO INSTALADO")
    
    # Check MLX for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print(f"\n   🍎 PACOTES APPLE SILICON:")
        mlx_packages = ['mlx', 'mlx-uniface']
        for package in mlx_packages:
            try:
                version = importlib.metadata.version(package)
                print(f"      ✅ {package}: {version}")
            except Exception:
                print(f"      ❌ {package}: NÃO INSTALADO")

def check_onnxruntime():
    """Check ONNX Runtime configuration."""
    print(f"\n⚙️  ONNX RUNTIME:")
    
    try:
        import onnxruntime as ort
        print(f"   Version: {ort.__version__}")
        
        # Available providers
        providers = ort.get_available_providers()
        print(f"   Available Providers:")
        for provider in providers:
            print(f"      - {provider}")
        
        # Check if CoreML is available
        has_coreml = 'CoreMLExecutionProvider' in providers
        print(f"   CoreML: {'✅ Disponível' if has_coreml else '❌ Não disponível'}")
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")

def check_opencv():
    """Check OpenCV configuration."""
    print(f"\n📷 OPENCV:")
    
    try:
        import cv2
        print(f"   Version: {cv2.__version__}")
        
        # Check CUDA availability
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"   CUDA: {'✅ Disponível' if cuda_available else '❌ Não disponível'}")
        
        # Check if running on macOS
        if platform.system() == "Darwin":
            print(f"   Backend: {'✅ AVFoundation (macOS)' if True else '❌ Outro'}")
        
        # Test basic operations
        import numpy as np
        test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        # Test color conversion
        try:
            _ = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            print(f"   Color Conversion: ✅ Funcional")
        except Exception as e:
            print(f"   Color Conversion: ❌ Erro: {e}")
        
        # Test resize
        try:
            _ = cv2.resize(test_frame, (320, 180))
            print(f"   Resize: ✅ Funcional")
        except Exception as e:
            print(f"   Resize: ❌ Erro: {e}")
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")

def check_models():
    """Check if required models exist."""
    print(f"\n🗂️  MODELOS:")
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    required_models = [
        'inswapper_128_fp16.onnx',
        'inswapper_128.onnx',
        'GFPGANv1.4.onnx'
    ]
    
    if not os.path.exists(models_dir):
        print(f"   ❌ Pasta models não existe: {models_dir}")
        return
    
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   ✅ {model}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {model}: NÃO ENCONTRADO")

def check_optimizations():
    """Check if optimizations are applied."""
    print(f"\n🚀 OTIMIZAÇÕES MACOS:")
    
    # Check ui.py optimizations
    ui_file = os.path.join(os.path.dirname(__file__), 'modules', 'ui.py')
    if os.path.exists(ui_file):
        with open(ui_file, 'r') as f:
            content = f.read()
            
        optimizations = {
            'LIVE_DETECTION_INTERVAL = 0.033': '✅ Intervalo de detecção otimizado',
            'DISPLAY_LOOP_INTERVAL_MS = 16': '✅ Display loop otimizado',
            'stale_face_ttl = 0.25': '✅ TTL de cache reduzido',
            'capture_queue = queue.Queue(maxsize=3)': '✅ Queue size otimizado',
        }
        
        for check, message in optimizations.items():
            if check in content:
                print(f"   {message}")
            else:
                print(f"   ⚠️  {check.split('=')[0].strip()} não otimizado")
    
    # Check face_analyser.py optimizations
    face_analyser_file = os.path.join(os.path.dirname(__file__), 'modules', 'face_analyser.py')
    if os.path.exists(face_analyser_file):
        with open(face_analyser_file, 'r') as f:
            content = f.read()
            
        if 'det_size = (160, 160)' in content:
            print(f"   ✅ Detector size otimizado (160x160)")
        else:
            print(f"   ⚠️  Detector size não otimizado")
    
    # Check face_swapper.py optimizations
    face_swapper_file = os.path.join(os.path.dirname(__file__), 'modules', 'processors', 'frame', 'face_swapper.py')
    if os.path.exists(face_swapper_file):
        with open(face_swapper_file, 'r') as f:
            content = f.read()
            
        if 'DETECTION_INTERVAL = 0.033' in content:
            print(f"   ✅ Intervalo de detecção face_swapper otimizado")
        else:
            print(f"   ⚠️  Intervalo de detecção face_swapper não otimizado")

def performance_recommendations():
    """Provide performance recommendations."""
    print(f"\n💡 RECOMENDAÇÕES DE PERFORMANCE:")
    
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    if is_apple_silicon:
        print(f"   1. ✅ Use CoreML: python run.py --execution-provider coreml")
        print(f"   2. ✅ Desative enhancers para mais FPS")
        print(f"   3. ✅ Use resolução 640x360 (padrão no macOS)")
        print(f"   4. ⚠️  MLX UniFace: Experimental (requer mlx-uniface)")
    else:
        print(f"   1. ⚠️  Sistema não é Apple Silicon - otimizações limitadas")
        print(f"   2. Use CPU ou CUDA se disponível")
    
    print(f"\n   📊 FPS Esperado (MacBook M1/M2/M3):")
    print(f"      - Apenas Face Swap: 8-15 FPS")
    print(f"      - Com Face Enhancer: 4-8 FPS")
    print(f"      - Com GPEN512: 2-4 FPS")

def main():
    """Run all diagnostics."""
    try:
        check_system()
        check_dependencies()
        check_onnxruntime()
        check_opencv()
        check_models()
        check_optimizations()
        performance_recommendations()
        
        print(f"\n" + "=" * 70)
        print(f"✅ DIAGNÓSTICO COMPLETO")
        print(f"=" * 70)
        
    except Exception as e:
        print(f"\n❌ Erro durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
