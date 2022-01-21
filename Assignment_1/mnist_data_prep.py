import numpy as np
import idx2numpy
from PIL import Image


def get_mnist_data_labels():
    train_image_file = "Assignment_1/data/mnist/train-images.idx3-ubyte"
    train_images = idx2numpy.convert_from_file(train_image_file)
    tid = train_images.shape
    # train_images_flattened = train_images.reshape(60000, 784)
    train_images_flattened = train_images.reshape(tid[0], tid[1] * tid[2])
    # image = Image.fromarray(train_images[12313])
    # image.show()

    train_label_file = "Assignment_1/data/mnist/train-labels.idx1-ubyte"
    train_labels = idx2numpy.convert_from_file(train_label_file)

    test_image_file = "Assignment_1/data/mnist/t10k-images.idx3-ubyte"
    test_images = idx2numpy.convert_from_file(test_image_file)
    tidt = test_images.shape
    test_images_flattened = test_images.reshape(tidt[0], tidt[1] * tidt[2])

    test_label_file = "Assignment_1/data/mnist/t10k-labels.idx1-ubyte"
    test_labels = idx2numpy.convert_from_file(test_label_file)

    return train_images_flattened, train_labels, test_images_flattened, test_labels
