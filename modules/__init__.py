import os 
import cv2
import numpy as np

# Utility function to support unicode characters in file paths for reading
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

# Utility function to support unicode characters in file paths for writing
def imwrite_unicode(path, img, params=None):
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
    result, encoded_img = cv2.imencode(ext, img, params if params else [])
    if result:
        encoded_img.tofile(path)
        return True
    return False