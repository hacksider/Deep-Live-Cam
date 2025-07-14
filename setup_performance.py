#!/usr/bin/env python3
"""
Deep-Live-Cam Performance Setup Script
Easy configuration for optimal performance based on your hardware
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.performance_manager import performance_manager
import psutil
import platform

def print_header():
    print("=" * 60)
    print("🎭 Deep-Live-Cam Performance Optimizer")
    print("=" * 60)
    print()

def analyze_system():
    """Analyze system specifications"""
    print("📊 Analyzing your system...")
    print("-" * 40)
    
    # System info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("GPU: Not available or not CUDA-compatible")
    except ImportError:
        print("GPU: PyTorch not available")
    
    print()

def show_performance_modes():
    """Display available performance modes"""
    print("🎯 Available Performance Modes:")
    print("-" * 40)
    
    modes = performance_manager.get_all_modes()
    for mode_name, mode_config in modes.items():
        print(f"\n{mode_name.upper()}:")
        print(f"  Quality Level: {mode_config['quality_level']}")
        print(f"  Target FPS: {mode_config['target_fps']}")
        print(f"  Detection Interval: {mode_config['face_detection_interval']}s")
        if 'description' in mode_config:
            print(f"  Description: {mode_config['description']}")

def interactive_setup():
    """Interactive performance setup"""
    print("🛠️ Interactive Setup:")
    print("-" * 40)
    
    print("\nChoose your priority:")
    print("1. Maximum FPS (for live streaming)")
    print("2. Balanced performance and quality")
    print("3. Best quality (for video processing)")
    print("4. Auto-optimize based on hardware")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                performance_manager.set_performance_mode("fast")
                print("✅ Set to FAST mode - Maximum FPS")
                break
            elif choice == "2":
                performance_manager.set_performance_mode("balanced")
                print("✅ Set to BALANCED mode - Good balance")
                break
            elif choice == "3":
                performance_manager.set_performance_mode("quality")
                print("✅ Set to QUALITY mode - Best results")
                break
            elif choice == "4":
                optimal_mode = performance_manager.optimize_for_hardware()
                print(f"✅ Auto-optimized to {optimal_mode.upper()} mode")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Setup cancelled.")
            return

def show_tips():
    """Show performance tips"""
    print("\n💡 Performance Tips:")
    print("-" * 40)
    
    tips = performance_manager.get_performance_tips()
    for tip in tips:
        print(f"  {tip}")

def main():
    print_header()
    analyze_system()
    show_performance_modes()
    interactive_setup()
    show_tips()
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete! You can change these settings anytime by running this script again.")
    print("💻 Start Deep-Live-Cam with: python run.py")
    print("=" * 60)

if __name__ == "__main__":
    main()