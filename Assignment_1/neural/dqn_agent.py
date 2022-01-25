ACTION_COUNT = 1
from Assignment_1.neural.dqn_network import DQN
import torch as t
import numpy as np


class DQNAgent:
    def __init__(self, **kwargs):
        self.model = DQN(**kwargs)
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

        self.batch_size = 64

    def update(self, state, action, reward, next_state, is_done):
        self.memory.store(state, action, reward, next_state, is_done)

    def get_training_batch_indexes(self):
        set_size = len(self.training_data_tensor)

        random_indexes = np.arange(set_size)
        np.random.shuffle(random_indexes)
        print(random_indexes)
        start_index = 0
        index_batches = []
        while start_index < set_size:
            end_index = start_index + self.batch_size
            # print(f"{start_index}:{end_index}")
            index_batches.append([random_indexes[start_index:end_index]])
            start_index = end_index

        return index_batches

    def test(self):
        self.model.eval()
        randindex = np.random.randint(0, len(self.training_data_tensor), 25)
        computed_raw = self.model.forward(self.training_data_tensor[randindex])
        computed_labels = t.where(computed_raw < 0.5, 0, 1)
        # computed_labels = t.round(t.sigmoid(computed_labels))
        for index, i in enumerate(randindex):
            print(f"Actual:{self.training_labels_tensor[i]} vs {computed_labels[index][0]}")

        print(f"Accuracy:{self.accuracy(computed_labels,self.training_labels_tensor[randindex]) * 100.0 :5.2f}% ")

    def accuracy(self, actual, computed):
        correct = (actual == computed).sum().float()
        acc = 100 * correct / actual.shape[0]
        return acc

    def test_set_accuracy(self):
        self.model.eval()
        with t.no_grad():
            computed_raw = self.model.forward(self.test_data_tensor)
            computed_labels = t.where(computed_raw < 0.5, 0, 1)
            # computed_labels = t.round(t.sigmoid(computed_labels))

        print(f"Test Accuracy:{self.accuracy(computed_labels,self.test_labels_tensor):5.2f}% ")

    def train(self):
        self.model.train()
        self.model.optimizer.zero_grad()

        batch_indexes = self.get_training_batch_indexes()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch in batch_indexes:
            computed_raw = self.model.forward(self.training_data_tensor[batch])
            computed_labels = t.where(computed_raw < 0.5, 0.0, 1.0)
            # print(f"{computed_raw[0]}  {computed_labels[0]}")
            loss = self.model.loss(self.training_labels_tensor[batch], computed_labels).to(self.model.device)
            loss.backward()

            step_loss = loss.item()
            epoch_loss += step_loss

            step_accuracy = self.accuracy(self.training_labels_tensor[batch], computed_labels)
            epoch_accuracy += step_accuracy
            self.model.optimizer.step()
            # print(f"Step loss:{step_loss:8.2f} accuracy:{step_accuracy:6.2f}%")
        epoch_loss = epoch_loss / len(batch_indexes)
        epoch_accuracy = epoch_accuracy / len(batch_indexes)
        print("============================================================")
        print(f"Epoch loss:{epoch_loss:8.2f} accuracy:{epoch_accuracy:6.2f}%")
        self.test_set_accuracy()
