# Add a function to handle GUI initialization with fallbacks
def initialize_gui():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Don't show until ready
        return root
    except Exception as e:
        print(f"GUI initialization failed: {e}")
        return None

# Add this to the main processing loop
def process_frame_with_gui_fallback(frame):
    try:
        # Process the frame normally
        return process_frame(frame)
    except Exception as e:
        print(f"Processing error: {e}")
        # Try to show a warning in the GUI if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Processing Error", str(e))
        except:
            pass  # Silently fail if GUI is not available
        return frame