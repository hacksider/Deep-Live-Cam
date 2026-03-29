#!/usr/bin/env python3

import os

# Prevent dependency-level update checks during startup.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Import the tkinter fix to patch the ScreenChanged error
import tkinter_fix

import core

if __name__ == '__main__':
    core.run()
