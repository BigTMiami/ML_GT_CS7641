from Assignment_1.neural.mnist_network_model import MNISTNet, MNISTData
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from torch.utils.data import DataLoader

(
    train_images_flattened,
    train_one_hot_labels,
    train_labels,
    test_images_flattened,
    test_one_hot_labels,
    test_labels,
) = get_mnist_data_labels_neural()

mnist = MNISTData(train_images_flattened, train_one_hot_labels)
mnist_loader = DataLoader(mnist, batch_size=100, shuffle=True)

net = MNISTNet(training_data_loader=mnist_loader, test_data=test_images_flattened, test_labels=test_labels)
net.check_test_accuracy()
results = net.train()


net.check_accuracy(test_images_flattened, test_labels)
