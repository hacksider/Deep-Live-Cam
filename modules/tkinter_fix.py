# Additional Tkinter compatibility handling for macOS
import platform
import os

def check_tkinter_support():
    if platform.system() == "Darwin":  # macOS
        # Check if we're running in a proper GUI environment
        if 'DISPLAY' not in os.environ and not os.environ.get('DISPLAY'):
            # Try to use a different backend or fallback
            try:
                import tkinter as tk
                return True
            except ImportError:
                return False
        return True
    return True