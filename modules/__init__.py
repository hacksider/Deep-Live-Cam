import os
import cv2
import numpy as np


# Utility function to support unicode characters in file paths for reading.
# OpenCV's cv2.imread() encodes the path with the locale ANSI code page on
# Windows, so it silently returns None for paths containing non-ASCII
# characters (Chinese, Japanese, Cyrillic, accents, ...). Reading the bytes
# through NumPy (which uses Python's unicode-aware file I/O) and decoding them
# in memory sidesteps that limitation. Returns None on failure, matching
# cv2.imread() so it stays a drop-in replacement.
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


# Utility function to support unicode characters in file paths for writing.
# cv2.imwrite() has the same ANSI-path limitation, so we encode the image in
# memory and write the bytes out with NumPy's unicode-aware file I/O. Returns
# True/False like cv2.imwrite() so it stays a drop-in replacement.
def imwrite_unicode(path, img, params=None):
    try:
        root, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"
        result, encoded_img = cv2.imencode(ext, img, params if params is not None else [])
        if not result:
            return False
        encoded_img.tofile(path)
        return True
    except Exception:
        return False
