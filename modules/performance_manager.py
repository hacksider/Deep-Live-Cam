"""
Performance Manager for Deep-Live-Cam
Handles performance mode switching and optimization settings
"""
import json
import os
from typing import Dict, Any
import modules.globals
from modules.performance_optimizer import performance_optimizer


class PerformanceManager:
    def __init__(self):
        self.config_path = "performance_config.json"
        self.config = self.load_config()
        self.current_mode = "balanced"
    
    def load_config(self) -> Dict[str, Any]:
        """Load performance configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            print(f"Error loading performance config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default performance configuration"""
        return {
            "performance_modes": {
                "fast": {
                    "quality_level": 0.6,
                    "face_detection_interval": 0.2,
                    "target_fps": 30,
                    "frame_skip": 2,
                    "enable_caching": True,
                    "processing_resolution_scale": 0.7
                },
                "balanced": {
                    "quality_level": 0.85,
                    "face_detection_interval": 0.1,
                    "target_fps": 25,
                    "frame_skip": 1,
                    "enable_caching": True,
                    "processing_resolution_scale": 0.85
                },
                "quality": {
                    "quality_level": 1.0,
                    "face_detection_interval": 0.05,
                    "target_fps": 20,
                    "frame_skip": 1,
                    "enable_caching": False,
                    "processing_resolution_scale": 1.0
                }
            }
        }
    
    def set_performance_mode(self, mode: str) -> bool:
        """Set performance mode (fast, balanced, quality)"""
        try:
            if mode not in self.config["performance_modes"]:
                print(f"Invalid performance mode: {mode}")
                return False
            
            mode_config = self.config["performance_modes"][mode]
            self.current_mode = mode
            
            # Apply settings to performance optimizer
            performance_optimizer.quality_level = mode_config["quality_level"]
            performance_optimizer.detection_interval = mode_config["face_detection_interval"]
            performance_optimizer.target_fps = mode_config["target_fps"]
            
            # Apply to globals
            modules.globals.performance_mode = mode
            modules.globals.quality_level = mode_config["quality_level"]
            modules.globals.face_detection_interval = mode_config["face_detection_interval"]
            modules.globals.target_live_fps = mode_config["target_fps"]
            
            print(f"Performance mode set to: {mode}")
            return True
            
        except Exception as e:
            print(f"Error setting performance mode: {e}")
            return False
    
    def get_current_mode(self) -> str:
        """Get current performance mode"""
        return self.current_mode
    
    def get_mode_info(self, mode: str) -> Dict[str, Any]:
        """Get information about a specific performance mode"""
        return self.config["performance_modes"].get(mode, {})
    
    def get_all_modes(self) -> Dict[str, Any]:
        """Get all available performance modes"""
        return self.config["performance_modes"]
    
    def optimize_for_hardware(self) -> str:
        """Automatically select optimal performance mode based on hardware"""
        try:
            import psutil
            import torch
            
            # Check available RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # Check GPU availability
            has_gpu = torch.cuda.is_available()
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            
            # Determine optimal mode
            if has_gpu and ram_gb >= 8 and cpu_cores >= 8:
                optimal_mode = "quality"
            elif has_gpu and ram_gb >= 4:
                optimal_mode = "balanced"
            else:
                optimal_mode = "fast"
            
            self.set_performance_mode(optimal_mode)
            print(f"Auto-optimized for hardware: {optimal_mode} mode")
            print(f"  RAM: {ram_gb:.1f}GB, GPU: {has_gpu}, CPU Cores: {cpu_cores}")
            
            return optimal_mode
            
        except Exception as e:
            print(f"Error in hardware optimization: {e}")
            self.set_performance_mode("balanced")
            return "balanced"
    
    def get_performance_tips(self) -> list:
        """Get performance optimization tips"""
        tips = [
            "🚀 Use 'Fast' mode for maximum FPS during live streaming",
            "⚖️ Use 'Balanced' mode for good quality with decent performance",
            "🎨 Use 'Quality' mode for best results when processing videos",
            "💾 Close other applications to free up system resources",
            "🖥️ Use GPU acceleration when available (CUDA/DirectML)",
            "📹 Lower camera resolution if experiencing lag",
            "🔄 Enable frame caching for smoother playback",
            "⚡ Ensure good lighting for better face detection"
        ]
        return tips


# Global performance manager instance
performance_manager = PerformanceManager()