import cv2
import numpy as np

# Define a new function that supports unicode characters in file paths
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

# Override the original `cv2.imread`
cv2.imread = imread_unicode

# Define a function to support unicode characters in file paths when saving
def imwrite_unicode(path, img, params=None):
    # Encode the image
    ext = path.split(".")[-1]  # Get file extension
    result, encoded_img = cv2.imencode(f".{ext}", img, params if params else [])

    if result:
        encoded_img.tofile(path)  # Save image using numpy's `tofile()`
        return True
    return False

# Override `cv2.imwrite`
cv2.imwrite = imwrite_unicode
