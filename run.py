#!/usr/bin/env python3

# Import the tkinter fix to patch the ScreenChanged error
import modules.tkinter_fix  # noqa: F401

from modules import core

if __name__ == '__main__':
    core.run()
