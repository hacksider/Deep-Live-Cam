import tkinter

# Only needs to be imported once at the beginning of the application
def apply_patch():
    # Create a monkey patch for the internal _tkinter module
    original_init = tkinter.Tk.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call the original init
        original_init(self, *args, **kwargs)
        
        # Define the missing ::tk::ScreenChanged procedure
        self.tk.eval("""
        if {[info commands ::tk::ScreenChanged] == ""} {
            proc ::tk::ScreenChanged {args} {
                # Do nothing
                return
            }
        }
        """)
    
    # Apply the monkey patch
    tkinter.Tk.__init__ = patched_init

# Apply the patch automatically when this module is imported
apply_patch() 