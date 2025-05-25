#!/usr/bin/env python3

from modules import core
import os
os.system('Xvfb :1 -screen 0 1600x1200x16 &')
os.environ['DISPLAY'] = ':1.0'

if __name__ == '__main__':
    core.run()
