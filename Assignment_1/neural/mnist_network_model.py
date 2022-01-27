from itertools import accumulate
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from torch.utils.data import Dataset, DataLoader


class MNISTData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MNISTNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        self.test_data = kwargs["test_data"]
        self.test_labels = kwargs["test_labels"]

        self.cv_data = kwargs["cv_data"]
        self.cv_labels = kwargs["cv_labels"]

        self.training_data_loader = kwargs["training_data_loader"]
        self.epoch_count = kwargs["epoch_count"]

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def check_accuracy(self, input, labels):
        output = self(input)
        _, predictions = t.max(output, 1)
        correct = (predictions == labels).sum().float()
        acc = 100 * correct / len(labels)
        return acc

    def check_test_accuracy(self):
        return self.check_accuracy(self.test_data, self.test_labels)

    def check_cv_accuracy(self):
        return self.check_accuracy(self.cv_data, self.cv_labels)

    def train(self):
        epoch_values = []
        for epoch in range(self.epoch_count):  # loop over the dataset multiple times
            running_loss = 0.0
            for inputs, labels in self.training_data_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            test_acc = self.check_test_accuracy()
            cv_acc = self.check_cv_accuracy()
            print(f"Epoch:{epoch} loss:{running_loss:10.4f} cv acc:{cv_acc:6.3f} test acc:{test_acc:6.3f}")
            epoch_values.append([epoch, running_loss, cv_acc, test_acc])
        return epoch_values


if False:
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

    net = MNISTNet()
    net.train(mnist_loader)

    print("Finished Training")

    net.check_accuracy(net, test_images_flattened, test_labels)
