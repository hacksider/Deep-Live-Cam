#!/usr/bin/env python3

# Import the tkinter fix to patch the ScreenChanged error (module patches Tk on import)
import tkinter_fix  # noqa: F401

import core

if __name__ == '__main__':
    core.run()
