
import cv2
import numpy as np

def resize_image(image, size=(256, 256)):
    """
    Resize image to target size using INTER_LINEAR interpolation.
    For masks, typically INTER_NEAREST is better to preserve integers, 
    so we might need a separate function or handled by caller.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def resize_mask(mask, size=(256, 256)):
    """Resize mask using INTER_NEAREST to preserve binary values."""
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Expects input image to be grayscale (single channel) and uint8.
    """
    if image.dtype != np.uint8:
        # Convert to uint8 assuming image is normalized 0-max or similar
        # If float 0-1, scale to 255.
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def normalize_to_01(image):
    """
    Normalize pixel values to [0, 1].
    Returns float32.
    """
    return image.astype(np.float32) / 255.0

def preprocess_pipeline_v1(image, mask=None, target_size=(256, 256)):
    """
    Full pipeline: Resize -> CLAHE (img) -> Normalize (output is float 0-1).
    Note: Returns float arrays.
    """
    # Resize
    img_resized = resize_image(image, target_size)
    if mask is not None:
        mask_resized = resize_mask(mask, target_size)
    
    # CLAHE
    img_clahe = apply_clahe(img_resized)
    
    # Normalize
    img_norm = normalize_to_01(img_clahe)
    if mask is not None:
        mask_norm = normalize_to_01(mask_resized) # Or keep as uint
        return img_norm, mask_norm
    
    return img_norm
