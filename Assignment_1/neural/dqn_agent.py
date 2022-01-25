ACTION_COUNT = 1
from Assignment_1.neural.dqn_network import DQN
from sklearn.model_selection import KFold

import torch as t
import numpy as np


class DQNAgent:
    def __init__(self, **kwargs):
        self.network_learning_rate = kwargs["network_learning_rate"]
        self.layer_1_count = kwargs["layer_one_size"]
        self.model = None
        self.load_model()

        self.training_data = kwargs["training_data"]
        self.training_data = (self.training_data).astype(np.float32)
        self.training_data_tensor = t.tensor(self.training_data, device=self.model.device)
        self.training_labels = kwargs["training_labels"]
        self.training_labels = np.reshape(self.training_labels, (len(self.training_labels), 1)).astype(np.float32)
        self.training_labels_tensor = t.tensor(self.training_labels, device=self.model.device)

        self.test_data = kwargs["test_data"]
        self.test_data = (self.test_data).astype(np.float32)
        self.test_data_tensor = t.tensor(self.test_data, device=self.model.device)
        self.test_labels = kwargs["test_labels"]
        self.test_labels = np.reshape(self.test_labels, (len(self.test_labels), 1)).astype(np.float32)
        self.test_labels_tensor = t.tensor(self.test_labels, device=self.model.device)

        self.cv_training_indexes = None
        self.cv_test_indexes = None

        self.batch_size = 64
        self.epoch_count = kwargs["epoch_count"]

    def load_model(self):
        self.model = DQN(network_learning_rate=self.network_learning_rate, layer_one_size=self.layer_1_count)

    def get_training_batch_indexes(self, index_set=None):
        if index_set is None:
            index_set = np.arange(len(self.training_data_tensor))
        set_size = len(index_set)

        np.random.shuffle(index_set)
        start_index = 0
        index_batches = []
        while start_index < set_size:
            end_index = start_index + self.batch_size
            index_batches.append([index_set[start_index:end_index]])
            start_index = end_index

        return index_batches

    def test(self):
        self.model.eval()
        randindex = np.random.randint(0, len(self.training_data_tensor), 100)
        computed_raw = self.model.forward(self.training_data_tensor[randindex])
        computed_labels = t.where(computed_raw < 0.5, 0, 1)
        # computed_labels = t.round(t.sigmoid(computed_labels))
        for index, i in enumerate(randindex):
            print(
                f"Actual:{self.training_labels_tensor[i]} vs {computed_labels[index][0]} {self.training_labels_tensor[i] == computed_labels[index][0]} {computed_raw[index][0]:10.3f}"
            )

        print(f"Accuracy:{self.accuracy(computed_labels,self.training_labels_tensor[randindex]) :5.2f}% ")

    def accuracy(self, actual, computed):
        correct = (actual == computed).sum().float()
        acc = 100 * correct / actual.shape[0]
        return acc

    def test_set_accuracy(self, show_details=False):
        self.model.eval()
        with t.no_grad():
            computed_raw = self.model.forward(self.test_data_tensor)
            computed_labels = t.where(computed_raw < 0.5, 0, 1)
            # computed_labels = t.round(t.sigmoid(computed_labels))

        acc = self.accuracy(computed_labels, self.test_labels_tensor)
        if show_details:
            print(f"Test Accuracy:{acc:5.2f}% ")

        return acc

    def cv_accuracy(self, cv_test_indexes):
        self.model.eval()
        with t.no_grad():
            computed_raw = self.model.forward(self.training_data_tensor[cv_test_indexes])
            computed_labels = t.where(computed_raw < 0.5, 0, 1)
            # computed_labels = t.round(t.sigmoid(computed_labels))

        acc = self.accuracy(computed_labels, self.training_labels_tensor[cv_test_indexes])
        return acc

    def train_with_cv(self, show_details=True):
        all_indexes = list(range(len(self.training_data_tensor)))
        kf = KFold(n_splits=4)
        split_count = 0
        epoch_values = np.empty((self.epoch_count, 4))
        for train_indexes, test_indexes in kf.split(all_indexes):
            print("============================================================")
            print(f"Split {split_count}")
            print("============================================================")
            self.load_model()
            epoch_values = self.train(
                show_details=show_details,
                cv_train_indexes=train_indexes,
                cv_test_indexes=test_indexes,
                epoch_values=epoch_values,
            )

            split_count += 1

        epoch_values = epoch_values / split_count

        return epoch_values

    def train(self, show_details=True, cv_train_indexes=None, cv_test_indexes=None, epoch_values=None):
        print("============================================================")
        print(f"Epoch   Loss      Acc      Val        CV")
        if epoch_values is None:
            epoch_values = np.empty((self.epoch_count, 4))
        cv_acc = 0.0
        for epoch_index in range(self.epoch_count):
            batch_indexes = self.get_training_batch_indexes(index_set=cv_train_indexes)
            epoch_loss, epoch_accuracy = self.train_epoch(batch_indexes)

            if cv_test_indexes is not None:
                cv_acc = self.cv_accuracy(cv_test_indexes)
            validation_accuracy = self.test_set_accuracy()
            epoch_values[epoch_index] += epoch_loss, epoch_accuracy, validation_accuracy, cv_acc
            if show_details:
                print(
                    f"{epoch_index:3}  {epoch_loss:8.4f}   {epoch_accuracy:7.3f} {validation_accuracy:7.3f}%  {cv_acc:7.3f}%"
                )

        return epoch_values

    def train_epoch(self, batch_indexes):
        self.model.train()
        self.model.optimizer.zero_grad()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch in batch_indexes:
            computed_raw = self.model.forward(self.training_data_tensor[batch])
            computed_labels = t.where(computed_raw < 0.5, 0.0, 1.0)
            loss = self.model.loss(self.training_labels_tensor[batch], computed_raw).to(self.model.device)
            loss.backward()

            step_loss = loss.item()
            epoch_loss += step_loss

            step_accuracy = self.accuracy(self.training_labels_tensor[batch], computed_labels)
            epoch_accuracy += step_accuracy
            self.model.optimizer.step()

        epoch_loss = epoch_loss / len(batch_indexes)
        epoch_accuracy = epoch_accuracy / len(batch_indexes)

        return epoch_loss, epoch_accuracy
