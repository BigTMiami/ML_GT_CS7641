import numpy as np
import idx2numpy
from PIL import Image
import torch as t

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


def int_to_one_hot(initial_array):
    return np.eye(10)[initial_array]


def get_mnist_data_labels(one_hot_labels=False, scale_data=False, as_tensor=False):
    train_image_file = "Assignment_1/data/mnist/train-images.idx3-ubyte"
    train_images = idx2numpy.convert_from_file(train_image_file)
    tid = train_images.shape
    # train_images_flattened = train_images.reshape(60000, 784)
    train_images_flattened = train_images.reshape(tid[0], tid[1] * tid[2])
    if scale_data:
        train_images_flattened = train_images_flattened / 255.0
    # image = Image.fromarray(train_images[12313])
    # image.show()

    train_label_file = "Assignment_1/data/mnist/train-labels.idx1-ubyte"
    train_labels = idx2numpy.convert_from_file(train_label_file)
    if one_hot_labels:
        train_labels = int_to_one_hot(train_labels)

    test_image_file = "Assignment_1/data/mnist/t10k-images.idx3-ubyte"
    test_images = idx2numpy.convert_from_file(test_image_file)
    tidt = test_images.shape
    test_images_flattened = test_images.reshape(tidt[0], tidt[1] * tidt[2])
    if scale_data:
        test_images_flattened = test_images_flattened / 255.0

    test_label_file = "Assignment_1/data/mnist/t10k-labels.idx1-ubyte"
    test_labels = idx2numpy.convert_from_file(test_label_file)
    if one_hot_labels:
        test_labels = int_to_one_hot(test_labels)

    if as_tensor:
        train_images_flattened = t.tensor(train_images_flattened.astype(np.float32))
        train_labels = t.tensor(train_labels.astype(np.float32))
        test_images_flattened = t.tensor(test_images_flattened.astype(np.float32))
        test_labels = t.tensor(test_labels.astype(np.float32))

    return train_images_flattened, train_labels, test_images_flattened, test_labels


def get_mnist_data_labels_neural():
    train_image_file = "Assignment_1/data/mnist/train-images.idx3-ubyte"
    train_images = idx2numpy.convert_from_file(train_image_file)
    tid = train_images.shape
    train_images_flattened = train_images.reshape(tid[0], tid[1] * tid[2])
    train_images_flattened = train_images_flattened / 255.0

    train_label_file = "Assignment_1/data/mnist/train-labels.idx1-ubyte"
    train_labels = idx2numpy.convert_from_file(train_label_file)
    train_one_hot_labels = int_to_one_hot(train_labels)

    test_image_file = "Assignment_1/data/mnist/t10k-images.idx3-ubyte"
    test_images = idx2numpy.convert_from_file(test_image_file)
    tidt = test_images.shape
    test_images_flattened = test_images.reshape(tidt[0], tidt[1] * tidt[2])
    test_images_flattened = test_images_flattened / 255.0

    test_label_file = "Assignment_1/data/mnist/t10k-labels.idx1-ubyte"
    test_labels = idx2numpy.convert_from_file(test_label_file)
    test_one_hot_labels = int_to_one_hot(test_labels)

    train_images_flattened = t.tensor(train_images_flattened.astype(np.float32))
    train_one_hot_labels = t.tensor(train_one_hot_labels.astype(np.float32))
    train_labels = t.tensor(train_labels.astype(np.float32))
    test_images_flattened = t.tensor(test_images_flattened.astype(np.float32))
    test_one_hot_labels = t.tensor(test_one_hot_labels.astype(np.float32))
    test_labels = t.tensor(test_labels.astype(np.float32))

    return (
        train_images_flattened,
        train_one_hot_labels,
        train_labels,
        test_images_flattened,
        test_one_hot_labels,
        test_labels,
    )
