
import os
import sys
import numpy as np
import tensorflow as tf

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
import unet_model

def verify_model():
    print("Verifying U-Net Model...")
    
    try:
        model = unet_model.build_unet(input_shape=(256, 256, 1))
        print("Model built successfully.")
        
        # Compile model to check loss function
        model.compile(optimizer='adam', loss=unet_model.dice_loss, metrics=[unet_model.dice_coef])
        print("Model compiled successfully with Dice Loss.")
        
        # Summary
        # model.summary() # Optional, can be verbose
        
        # Test input shape
        inputs = np.random.rand(1, 256, 256, 1).astype(np.float32)
        print(f"Testing with input shape: {inputs.shape}")
        
        # Predict
        outputs = model.predict(inputs)
        print(f"Prediction output shape: {outputs.shape}")
        
        if outputs.shape == (1, 256, 256, 1):
            print("Verification SUCCESS: Output shape matches expected (Batch, 256, 256, 1).")
            return True
        else:
            print(f"Verification FAILED: Output shape mismatch. Expected (1, 256, 256, 1), got {outputs.shape}")
            return False
            
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        return False

if __name__ == "__main__":
    verify_model()
