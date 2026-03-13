
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient metric.
    Calculates overlap between ground truth and prediction.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice Loss function.
    Minimizes 1 - Dice Coefficient.
    """
    return 1.0 - dice_coef(y_true, y_pred)

def conv_block(input_tensor, num_filters):
    """
    Standard encoder block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """
    Encoder block + MaxPool
    """
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """
    Decoder block: UpSample -> Concatenate -> Conv Block
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 1)):
    """
    Builds the U-Net model.
    4 Downsampling blocks, 4 Upsampling blocks.
    """
    inputs = layers.Input(input_shape)
    
    # Encoder
    x1, p1 = encoder_block(inputs, 64)
    x2, p2 = encoder_block(p1, 128)
    x3, p3 = encoder_block(p2, 256)
    x4, p4 = encoder_block(p3, 512)
    
    # Bridge / Bottleneck
    b1 = conv_block(p4, 1024)
    
    # Decoder
    d1 = decoder_block(b1, x4, 512)
    d2 = decoder_block(d1, x3, 256)
    d3 = decoder_block(d2, x2, 128)
    d4 = decoder_block(d3, x1, 64)
    
    # Output
    # Binary segmentation: Sigmoid activation
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
    
    model = models.Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()
