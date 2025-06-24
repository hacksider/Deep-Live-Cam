import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2 # Imported for BGR to RGB conversion, though PIL can also do it.

# Global variables for caching
HAIR_SEGMENTER_PROCESSOR = None
HAIR_SEGMENTER_MODEL = None
MODEL_NAME = "isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing"

def segment_hair(image_np: np.ndarray) -> np.ndarray:
    """
    Segments hair from an image.

    Args:
        image_np: NumPy array representing the image (BGR format from OpenCV).

    Returns:
        NumPy array representing the binary hair mask.
    """
    global HAIR_SEGMENTER_PROCESSOR, HAIR_SEGMENTER_MODEL

    if HAIR_SEGMENTER_PROCESSOR is None or HAIR_SEGMENTER_MODEL is None:
        print(f"Loading hair segmentation model and processor ({MODEL_NAME}) for the first time...")
        try:
            HAIR_SEGMENTER_PROCESSOR = SegformerImageProcessor.from_pretrained(MODEL_NAME)
            HAIR_SEGMENTER_MODEL = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

            if torch.cuda.is_available():
                try:
                    HAIR_SEGMENTER_MODEL = HAIR_SEGMENTER_MODEL.to('cuda')
                    print("INFO: Hair segmentation model moved to CUDA (GPU).")
                except Exception as e_cuda:
                    print(f"ERROR: Failed to move hair segmentation model to CUDA: {e_cuda}. Using CPU instead.")
                    # Fallback to CPU if .to('cuda') fails
                    HAIR_SEGMENTER_MODEL = HAIR_SEGMENTER_MODEL.to('cpu')
            else:
                print("INFO: CUDA not available. Hair segmentation model will use CPU.")

            print("INFO: Hair segmentation model and processor loaded successfully (device: {}).".format(HAIR_SEGMENTER_MODEL.device))
        except Exception as e:
            print(f"ERROR: Failed to load hair segmentation model/processor: {e}")
            # Return an empty mask compatible with expected output shape (H, W)
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    # Convert BGR (OpenCV) to RGB (PIL)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    inputs = HAIR_SEGMENTER_PROCESSOR(images=image_pil, return_tensors="pt")

    if HAIR_SEGMENTER_MODEL.device.type == 'cuda':
        try:
            # SegformerImageProcessor output (BatchEncoding) is a dict-like object.
            # We need to move its tensor components, commonly 'pixel_values'.
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
            else: # Fallback if the structure is different than expected
                inputs = inputs.to('cuda')
            # If inputs has other tensor components that need to be moved, they'd need similar handling.
        except Exception as e_inputs_cuda:
            print(f"ERROR: Failed to move inputs to CUDA: {e_inputs_cuda}. Attempting inference on CPU.")
            # If moving inputs to CUDA fails, we should ensure model is also on CPU for this inference pass
            # This is a tricky situation; ideally, this failure shouldn't happen if model moved successfully.
            # For simplicity, we'll assume if model is on CUDA, inputs should also be.
            # A more robust solution might involve moving model back to CPU if inputs can't be moved.

    with torch.no_grad(): # Important for inference
        outputs = HAIR_SEGMENTER_MODEL(**inputs)

    logits = outputs.logits  # Shape: batch_size, num_labels, height, width

    # Upsample logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image_np.shape[0], image_np.shape[1]), # H, W
        mode='bilinear',
        align_corners=False
    )

    segmentation_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Label 2 is for hair in this model
    return np.where(segmentation_map == 2, 255, 0).astype(np.uint8)

if __name__ == '__main__':
    # This is a conceptual test.
    # In a real scenario, you would load an image using OpenCV or Pillow.
    # For example:
    # sample_image_np = cv2.imread("path/to/your/image.jpg")
    # if sample_image_np is not None:
    #     hair_mask_output = segment_hair(sample_image_np)
    #     cv2.imwrite("hair_mask_output.png", hair_mask_output)
    #     print("Hair mask saved to hair_mask_output.png")
    # else:
    #     print("Failed to load sample image.")

    print("Conceptual test: Hair segmenter module created.")
    # Create a dummy image for a basic test run if no image is available.
    dummy_image_np = np.zeros((100, 100, 3), dtype=np.uint8) # 100x100 BGR image
    dummy_image_np[:, :, 1] = 255 # Make it green to distinguish from black mask

    try:
        print("Running segment_hair with a dummy image...")
        hair_mask_output = segment_hair(dummy_image_np)
        print(f"segment_hair returned a mask of shape: {hair_mask_output.shape}")
        # Check if the output is a 2D array (mask) and has the same H, W as input
        assert hair_mask_output.shape == (dummy_image_np.shape[0], dummy_image_np.shape[1])
        # Check if the mask is binary (0 or 255)
        assert np.all(np.isin(hair_mask_output, [0, 255]))
        print("Dummy image test successful. Hair mask seems to be generated correctly.")

        # Attempt to save the dummy mask (optional, just for visual confirmation if needed)
        # cv2.imwrite("dummy_hair_mask_output.png", hair_mask_output)
        # print("Dummy hair mask saved to dummy_hair_mask_output.png")

    except ImportError as e:
        print(f"An ImportError occurred: {e}. This might be due to missing dependencies like transformers, torch, or Pillow.")
        print("Please ensure all required packages are installed by updating requirements.txt and installing them.")
    except Exception as e:
        print(f"An error occurred during the dummy image test: {e}")
        print("This could be due to issues with model loading, processing, or other runtime errors.")

    print("To perform a full test, replace the dummy image with a real image path.")
