#!/usr/bin/env python3

from modules import core
import sys
import os

# python embedded support: Ensure the script's folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    core.run()
