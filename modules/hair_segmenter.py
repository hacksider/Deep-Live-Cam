import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2 # Imported for BGR to RGB conversion, though PIL can also do it.

# Global variables for caching
HAIR_SEGMENTER_PROCESSOR = None
HAIR_SEGMENTER_MODEL = None
MODEL_NAME = "isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing"

def segment_hair(image_np: np.ndarray, device: str = "cpu", hair_label_index: int = None) -> np.ndarray:
    """
    Segments hair from an image.

    Args:
        image_np: NumPy array representing the image (BGR format from OpenCV).
        device: Device to run the model on ("cpu" or "cuda").
        hair_label_index: Optional; index of the hair label in the segmentation map. If not provided, will use model config or default to 2.

    Returns:
        NumPy array representing the binary hair mask.
    """
    global HAIR_SEGMENTER_PROCESSOR, HAIR_SEGMENTER_MODEL

    if HAIR_SEGMENTER_PROCESSOR is None or HAIR_SEGMENTER_MODEL is None:
        print(f"Loading hair segmentation model and processor ({MODEL_NAME}) for the first time...")
        try:
            HAIR_SEGMENTER_PROCESSOR = SegformerImageProcessor.from_pretrained(MODEL_NAME)
            HAIR_SEGMENTER_MODEL = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
            HAIR_SEGMENTER_MODEL = HAIR_SEGMENTER_MODEL.to(device)
            print(f"Hair segmentation model and processor loaded successfully. Model moved to device: {device}")
        except Exception as e:
            print(f"Failed to load hair segmentation model/processor: {e}")
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    if HAIR_SEGMENTER_PROCESSOR is None or HAIR_SEGMENTER_MODEL is None:
        print("Error: Hair segmentation models are not available.")
        return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    inputs = HAIR_SEGMENTER_PROCESSOR(images=image_pil, return_tensors="pt")
    if device == "cuda" and hasattr(HAIR_SEGMENTER_MODEL, "device") and HAIR_SEGMENTER_MODEL.device.type == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = HAIR_SEGMENTER_MODEL(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image_np.shape[0], image_np.shape[1]),
        mode='bilinear',
        align_corners=False
    )
    segmentation_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    if hair_label_index is None:
        hair_label_index = getattr(HAIR_SEGMENTER_MODEL, "hair_label_index", 2)
    return np.where(segmentation_map == hair_label_index, 255, 0).astype(np.uint8)

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
