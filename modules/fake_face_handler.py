import os
import requests
import tempfile
from pathlib import Path
import cv2
import numpy as np
import modules.globals

def add_padding_to_face(image, padding_ratio=0.3):
    """Add padding around the face image
    
    Args:
        image: The input face image
        padding_ratio: Amount of padding to add as a ratio of image dimensions
        
    Returns:
        Padded image with background padding added
    """
    if image is None:
        return None
        
    height, width = image.shape[:2]
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    
    # Create larger image with padding
    padded_height = height + 2 * pad_y
    padded_width = width + 2 * pad_x
    padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    
    # Fill padded area with blurred and darkened edge pixels
    edge_color = cv2.blur(image, (15, 15))
    edge_color = (edge_color * 0.6).astype(np.uint8)  # Darken the padding
    
    # Fill the padded image with original face
    padded_image[pad_y:pad_y+height, pad_x:pad_x+width] = image
    
    # Fill padding areas with edge color
    # Top padding - repeat first row
    top_edge = edge_color[0, :, :]
    for i in range(pad_y):
        padded_image[i, pad_x:pad_x+width] = top_edge
        
    # Bottom padding - repeat last row
    bottom_edge = edge_color[-1, :, :]
    for i in range(pad_y):
        padded_image[pad_y+height+i, pad_x:pad_x+width] = bottom_edge
        
    # Left padding - repeat first column
    left_edge = edge_color[:, 0, :]
    for i in range(pad_x):
        padded_image[pad_y:pad_y+height, i] = left_edge
        
    # Right padding - repeat last column
    right_edge = edge_color[:, -1, :]
    for i in range(pad_x):
        padded_image[pad_y:pad_y+height, pad_x+width+i] = right_edge
        
    # Fill corners with nearest edge colors
    # Top-left corner
    padded_image[:pad_y, :pad_x] = edge_color[0, 0, :]
    # Top-right corner
    padded_image[:pad_y, pad_x+width:] = edge_color[0, -1, :]
    # Bottom-left corner
    padded_image[pad_y+height:, :pad_x] = edge_color[-1, 0, :]
    # Bottom-right corner
    padded_image[pad_y+height:, pad_x+width:] = edge_color[-1, -1, :]
    
    return padded_image

def get_fake_face() -> str:
    """Fetch a face from thispersondoesnotexist.com and save it temporarily"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "deep-live-cam"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate temp file path
        temp_file = temp_dir / "fake_face.jpg"
        
        # Basic headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the image
        response = requests.get('https://thispersondoesnotexist.com', headers=headers)
        
        if response.status_code == 200:
            # Read image from response
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Add padding around the face
            padded_image = add_padding_to_face(image)
            
            # Save the padded image
            cv2.imwrite(str(temp_file), padded_image)
            return str(temp_file)
        else:
            print(f"Failed to fetch fake face: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching fake face: {str(e)}")
        return None

def cleanup_fake_face():
    """Clean up the temporary fake face image"""
    try:
        if modules.globals.fake_face_path and os.path.exists(modules.globals.fake_face_path):
            os.remove(modules.globals.fake_face_path)
            modules.globals.fake_face_path = None
    except Exception as e:
        print(f"Error cleaning up fake face: {str(e)}")

def refresh_fake_face():
    """Refresh the fake face image"""
    cleanup_fake_face()
    modules.globals.fake_face_path = get_fake_face()
    return modules.globals.fake_face_path is not None 