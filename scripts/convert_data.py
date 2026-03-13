
import os
import glob
import numpy as np
import h5py
import scipy.io
from PIL import Image

# Add current directory to path to import preprocessing if needed, or structured import
# For this script structure, local import works if running from scripts/ or root with python -m
import preprocessing

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed', 'images')
OUTPUT_MASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed', 'masks')

def normalize_image(image):
    """Normalize image to 0-255 range."""
    image = image.astype(float)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()
    image *= 255.0
    return image.astype(np.uint8)

def load_mat_file(filepath):
    """
    Load .mat file, handling both older versions (scipy) and v7.3 (h5py).
    Returns a dictionary with 'cjdata' content.
    """
    try:
        # Try loading with scipy (older versions)
        mat = scipy.io.loadmat(filepath)
        # cjdata in scipy is a structured array inside a cell or similar
        # Depending on how it's saved, getting the struct might vary slightly
        # Usually it's mat['cjdata']
        dataset = mat['cjdata']
        
        # Extract fields. Note: scipy loads structs as numpy structured arrays
        # The shape is usually (1, 1), so we access [0, 0]
        data = dataset[0, 0]
        
        return {
            'image': data['image'],
            'tumorMask': data['tumorMask'],
            'label': data['label'][0, 0], # Scalar
            'PID': data['PID'][0] if len(data['PID']) > 0 else ''
        }
    except NotImplementedError:
        # Fallback to h5py for v7.3 mat files
        with h5py.File(filepath, 'r') as f:
            # h5py structure is different. cjdata is a group.
            cjdata = f['cjdata']
            
            # Images in v7.3 mat files are often transposed when read into python
            # because MATLAB converts to column-major, h5py reads as row-major (or vice versa logic)
            # So transposing back is often needed for images.
            image = np.array(cjdata['image'])
            tumorMask = np.array(cjdata['tumorMask'])
            label = np.array(cjdata['label'])[0, 0]
            
            # Check dimensions to decide if transpose is needed. 
            # Usually we expect typical MRI dims like 512x512.
            
            return {
                'image': image,
                'tumorMask': tumorMask,
                'label': label,
                'PID': '' # PID reading from h5py variable string can be tricky, skipping for now as not strictly needed for image gen
            }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)
    if not os.path.exists(OUTPUT_MASKS_DIR):
        os.makedirs(OUTPUT_MASKS_DIR)

    mat_files = glob.glob(os.path.join(DATA_DIR, '*.mat'))
    print(f"Found {len(mat_files)} .mat files.")

    for i, filepath in enumerate(mat_files):
        data = load_mat_file(filepath)
        if data is None:
            continue
        
        image_data = data['image']
        mask_data = data['tumorMask']
        
        # Ensure 2D
        if image_data.ndim > 2:
            image_data = image_data.squeeze()
        if mask_data.ndim > 2:
            mask_data = mask_data.squeeze()
            

        # Normalize raw data to 0-255 uint8 first for CLAHE
        img_uint8 = normalize_image(image_data)
        mask_uint8 = normalize_image(mask_data)
        
        # Apply Preprocessing (Resize, CLAHE)
        # Note: We keep as 0-255 uint8 for PNG storage.
        # Normalization to [0, 1] usually happens at load time for training.
        
        img_proc = preprocessing.resize_image(img_uint8, (256, 256))
        img_proc = preprocessing.apply_clahe(img_proc)
        
        mask_proc = preprocessing.resize_mask(mask_uint8, (256, 256))
        
        filename = os.path.basename(filepath).replace('.mat', '')
        
        # Save image
        out_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"{filename}.png")
        Image.fromarray(img_proc).save(out_img_path)
        
        # Save mask
        out_mask_path = os.path.join(OUTPUT_MASKS_DIR, f"{filename}_mask.png")
        Image.fromarray(mask_proc).save(out_mask_path)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} files...")

    print("Conversion complete.")

if __name__ == "__main__":
    main()
