# Add a function to help with macOS compatibility
def setup_macos_environment():
    import platform
    if platform.system() == "Darwin":
        import os
        # Set environment variables for macOS
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # Additional macOS-specific setup
        try:
            import tkinter as tk
            # Use the more stable Aqua theme if available
            if hasattr(tk, 'Tk') and callable(getattr(tk.Tk, 'tk', None)):
                return True
        except ImportError:
            return False

# Add this function to handle GUI initialization with fallbacks
def initialize_gui_with_fallback():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        return root
    except ImportError:
        print("Tkinter not available - falling back to console mode")
        return None
    except Exception as e:
        print(f"GUI initialization error: {e}")
        return None