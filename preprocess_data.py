"""
A data preprocesser.
TODO Fix error on dtype conversion
"""
import numpy as np
import tensorflow as tf


def normalize(data, old_min=0, old_max=255):
    """
    Transform data with range [old_min, old_max] to [0, 1].
    """
    return np.divide((data - old_min), (old_max-old_min), dtype=np.float)

def grayscale(data, axis):
    """
    Transform and RGB image to grayscale image.
    """
    return int(np.mean(data, axis=axis))

def augment(image):
    """
    Perform data augmentation on given image.
    TODO Randomize augmentation more by using random combinations.
    """
    augmented_images = [ image ]

    # Rotation
    rotation_angle = np.random.random() / 9
    augmented_images.append(tf.image.rot90(image, k=rotation_angle))
    augmented_images.append(tf.image.rot90(image, k=-1 * rotation_angle))

    # Brightness
    delta = np.random.random() / 2
    augmented_images.append(tf.image.adjust_brightness(image, delta=delta))
    augmented_images.append(tf.image.adjust_brightness(image, delta=-1 * delta))

    # TODO Contrast, Hue,  Gamma, Saturation

    # Crop
    central_frac = np.random.random() / 10
    tf.image.central_crop(image, central_fraction=central_frac)

    return augmented_images
