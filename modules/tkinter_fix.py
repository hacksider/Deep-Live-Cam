import os
import sys
import glob


def _fix_tcl_tk_paths():
    """Set TCL_LIBRARY and TK_LIBRARY if not already set.

    On Windows, the standard Python installer puts Tcl/Tk under
    ``<prefix>/tcl/tcl8.x`` rather than ``<prefix>/lib/tcl8.x``,
    so _tkinter cannot find init.tcl without help.

    IMPORTANT: Must run BEFORE ``import tkinter`` / ``import _tkinter``,
    because the C extension reads TCL_LIBRARY at import time.
    """
    if os.environ.get("TCL_LIBRARY") and os.environ.get("TK_LIBRARY"):
        # Both are set to non-empty values — nothing to do
        return

    base = sys.base_prefix
    # Search common locations: <base>/tcl/tcl*/init.tcl and <base>/lib/tcl*/init.tcl
    for subdir in ("tcl", "lib"):
        candidates = glob.glob(os.path.join(base, subdir, "tcl*", "init.tcl"))
        if candidates:
            tcl_dir = os.path.dirname(candidates[0])
            if not os.environ.get("TCL_LIBRARY"):
                os.environ["TCL_LIBRARY"] = tcl_dir
            # Derive TK_LIBRARY from the sibling tk directory
            parent = os.path.dirname(tcl_dir)
            tk_dirs = glob.glob(os.path.join(parent, "tk*"))
            if tk_dirs and not os.environ.get("TK_LIBRARY"):
                os.environ["TK_LIBRARY"] = tk_dirs[0]
            break


# Fix Tcl/Tk paths BEFORE importing tkinter
_fix_tcl_tk_paths()

import tkinter  # noqa: E402


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