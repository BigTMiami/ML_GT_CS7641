ACTION_COUNT = 1
from Assignment_1.neural.dqn_network import DQN
import torch as t
import numpy as np


class DQNAgent:
    def __init__(self, **kwargs):
        self.model = DQN(**kwargs)
        self.training_data = kwargs["training_data"]
        self.training_data = (self.training_data / 255).astype(np.float32)
        # self.training_data_tensor = t.tensor(self.training_data).to(self.model.device)
        self.training_data_tensor = t.tensor(self.training_data, device=self.model.device)
        self.training_labels = kwargs["training_labels"]
        # self.training_labels_tensor = t.tensor(self.training_labels).to(self.model.device)
        self.training_labels_tensor = t.tensor(self.training_labels, device=self.model.device)

    def update(self, state, action, reward, next_state, is_done):
        self.memory.store(state, action, reward, next_state, is_done)

    def test(self):
        computed_labels = self.model.forward(self.training_data_tensor[:10])
        for i in range(10):
            print(f"Actual:{self.training_labels_tensor[i]} vs {computed_labels[i][0]}")

    def train(self):

        self.model.optimizer.zero_grad()

        # Need to index each row in the tensor (action_batch_index), then use the which action ( action_batch) I took
        computed_labels = self.model.forward(self.training_data_tensor[:10])

        loss = self.model.loss(self.training_labels_tensor[:10], computed_labels).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
