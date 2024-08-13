#!/usr/bin/env python3

from modules import core

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

if __name__ == '__main__':
    core.run()
