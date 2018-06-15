"""
A Tensorflow data loader for images and labels compressed in gzip format.
"""
import gzip
import numpy as np
import tensorflow as tf


class DataLoader:
    """
    A loader for a gzipped file to prepare data for preprocessing.
    """
    def __init__(self,
                 n_images,
                 image_h=28,
                 image_w=28,
                 n_channels=1,
                 pixel_depth=255):
        """
        :param n_images: Number of images in the compressed file.
        :param image_h: The height of the image in the compressed file.
        :param image_w: The width of the image in the compressed file.
        :param n_channels: The number of channels of the image in the compressed
            file.
        """
        self.N_IMAGES = n_images
        self.IMAGE_H = image_h
        self.IMAGE_W = image_w
        self.N_CHANNELS = n_channels
        self.PIXEL_DEPTH = pixel_depth

    def extract_data(self, filename):
        """
        Extract file with given filename and return extracted data.

        :param filename: Name of the file with gzip format containing images.
            The filename should include the extension.
        :returns: A list of 4D tensors (image_id, image_h, image_w, n_channels).
        """
        HEADER_SIZE = 16

        print ('[extract_data] Extracting gzipped data from ', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(HEADER_SIZE)
            buf = bytestream.read(self.IMAGE_H * self.IMAGE_W * self.N_IMAGES)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(self.N_IMAGES, self.IMAGE_H,
                                self.IMAGE_W, self.N_CHANNELS)

        print ('[extract_data] Finished extraction of ', filename)
        return data

    def extract_label(self, filename):
        """
        Extract the labels into a vector of np.int64 label ids.

        :param filename: Name of the file with gzip format containing labels.
            The filename should include the extension.
        """
        HEADER_SIZE = 8

        print ('[extract_label] Extracting gzipped data from ', filename)
        with gzip.open(filename=filename) as bytestream:
            bytestream.read(HEADER_SIZE)
            buf = bytestream.read(1 * self.N_IMAGES)
            # type cast from uint8 to np.int64 to work in tensorflow framework
            labels = np.frombuffer(buffer=buf, dtype=np.uint8).astype(np.int64)

        print ('[extract_label] Finished extraction of ', filename)
        return labels


class MNISTDataLoader(DataLoader):
    """
    A data loader for MNIST dataset by Yann LeCun.
    http://yann.lecun.com/exdb/mnist/
    """
    def __init__(self, n_images):
        """
        :param n_images: Number of images in the compressed file.
        """
        super().__init__(n_images=n_images,
                         image_h=28,
                         image_w=28,
                         n_channels=1,
                         pixel_depth=255)
