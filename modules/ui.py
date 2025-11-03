# This is the main UI module that likely needs to handle the Tkinter setup
import tkinter as tk
from tkinter import ttk
import sys
import warnings

# Suppress the Tkinter version warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Tk.*")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deep-Live-Cam")
        self.geometry("800x600")
        
        # Add some debug info in case of issues
        print("Tkinter version:", tk.TkVersion)
        
        # Try to use themed widgets for better compatibility
        self.style = ttk.Style()
        self.style.theme_use("clam")  # More modern look
        
        # Initialize the main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=both, expand=True)
        
        # Add some content to verify it's working
        self.label = ttk.Label(self.main_frame, text="Deep-Live-Cam is loading...")
        self.label.pack(pady=20)
        
        # Add a button to trigger the face processing
        self.button = ttk.Button(self.main_frame, text="Load Face", command=self.load_face)
        self.button.pack(pady=10)
        
    def load_face(self):
        # This would be where the face processing starts
        print("Loading face...")
        
    def show_error(self, message):
        # Show error messages in a dialog
        from tkinter import messagebox
        messagebox.showerror("Error", message)

# Add this function to handle the Tkinter setup with the environment variable
def setup_tkinter():
    import os
    # Set the environment variable to suppress the deprecation warning
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    # Try to use the newer themed widgets if available
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window until we're ready
        return root
    except Exception as e:
        print(f"Error initializing Tkinter: {e}")
        return None

# Add this to the main execution flow
if __name__ == "__main__":
    # First setup the environment
    import os
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    # Then try to create the UI
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"GUI Error: {e}")
        # Fallback to console mode if GUI fails
        print("Running in console mode...")
        # Continue with command line processing