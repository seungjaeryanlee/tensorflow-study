"""
Defines operations to reshape data to use it in the model.
"""

def int_to_onehot(int_label, label_size=10, dtype=np.uint8):
    """
    Reshape integer labels to one-hot vectors.
    """
    onehot_label = np.zeros(label_size, dtype=dtype)
    onehot_label[int_label] = 1
    
    return onehot_label

def reshape_image(image):
    """
    Reshape 2D image (height, width) to 3D image (height, width, channel).
    """
    return np.expand_dims(image, axis=2)
