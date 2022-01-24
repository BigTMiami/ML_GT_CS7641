import torch as t

LEARNING_RATE = 0.01
LAYER_1_SIZE = 40
INPUT_COUNT = 40
ACTION_COUNT = 1


class DQN(t.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__()
        self.network_learning_rate = LEARNING_RATE
        self.layer_1_count = LAYER_1_SIZE

        # Set up layers
        self.l1 = t.nn.Linear(INPUT_COUNT, self.layer_1_count)
        t.nn.init.xavier_uniform_(self.l1.weight)
        self.l3 = t.nn.Linear(self.layer_1_count, ACTION_COUNT)
        t.nn.init.xavier_uniform_(self.l3.weight)

        # Setup optimizer, loss, device
        self.optimizer = t.optim.Adam(self.parameters(), lr=self.network_learning_rate)
        self.loss = t.nn.MSELoss()
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state_tensor):
        temp = t.nn.functional.relu(self.l1(state_tensor))
        output = self.l3(temp)
        return output

    def save_model(self, filename=None):
        if not filename:
            filename = f"lr_{self.network_learning_rate}_{self.layer_1_count}_{self.layer_2_count}_{get_session_id()}"
        full_filename = "./models/" + filename + ".pt"
        self.eval()
        t.save(self.state_dict(), full_filename)
        return full_filename

    def load_model(self, filename):
        self.load_state_dict(t.load(filename))
        self.eval()
