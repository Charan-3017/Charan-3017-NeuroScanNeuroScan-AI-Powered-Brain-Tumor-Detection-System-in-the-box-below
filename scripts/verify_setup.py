
import os
import sys
import numpy as np
import scipy.io
from PIL import Image

# Add current directory to path to import convert_data
sys.path.append(os.path.dirname(__file__))
import convert_data

def create_mock_mat(filepath):
    print(f"Creating mock .mat file at {filepath}...")
    # Create random image and mask
    image = np.random.rand(512, 512)
    mask = np.random.randint(0, 2, (512, 512))
    label = 1
    PID = '123456'
    
    # Create struct structure for scipy.io.savemat
    # cjdata should be a struct. In Python dict for savemat:
    # {'cjdata': {'image': ..., 'tumorMask': ..., ...}}
    
    cjdata = {
        'image': image,
        'tumorMask': mask,
        'label': label,
        'PID': PID
    }
    
    scipy.io.savemat(filepath, {'cjdata': cjdata})
    print("Mock file created.")

def verify_conversion():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    mock_path = os.path.join(data_dir, 'mock_test.mat')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    create_mock_mat(mock_path)
    
    print("Running conversion...")
    try:
        convert_data.main()
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False
        
    # Check output
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed', 'images')
    masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed', 'masks')
    
    img_out = os.path.join(images_dir, 'mock_test.png')
    mask_out = os.path.join(masks_dir, 'mock_test_mask.png')
    
    if os.path.exists(img_out) and os.path.exists(mask_out):
        print("Output files generated.")
        
        # Verify dimensions
        with Image.open(img_out) as img:
            if img.size == (256, 256):
                print("Verification SUCCESS: Image size is 256x256.")
            else:
                print(f"Verification FAILED: Image size is {img.size}, expected (256, 256).")
                return False
        
        print(f"Image: {img_out}")
        print(f"Mask: {mask_out}")
        
        # Cleanup
        os.remove(mock_path)
        os.remove(img_out)
        os.remove(mask_out)
        print("Cleanup complete.")
        return True
    else:
        print("Verification FAILED: Output files missing.")
        return False

if __name__ == "__main__":
    verify_conversion()
