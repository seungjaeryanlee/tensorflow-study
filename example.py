#! /usr/bin/env python3
"""
A sample code to display how each module can be used.
"""

from data_loader import MNISTDataLoader

def main():
    loader = MNISTDataLoader(1)
    data = loader.extract_data('data/mnist/train-images-idx3-ubyte.gz')
    data = loader.rescale_data(data)
    labels = loader.extract_label('data/mnist/train-labels-idx1-ubyte.gz')

if __name__ == '__main__':
    main()
