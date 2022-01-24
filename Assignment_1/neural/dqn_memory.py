ACTION_COUNT = 1
INPUT_COUNT = 40
import numpy as np
import torch as t


class DQNMemory:
    def __init__(self, **kwargs):
        self.memory_max = kwargs["memory_max"]
        self.batch_size = kwargs["memory_batch_size"]
        self.mem_step = 0
        self.current_memory_size = 0
        self.batch_available = False

        self.state = np.zeros((self.memory_max, INPUT_COUNT), dtype=np.float32)
        self.next_state = np.zeros((self.memory_max, INPUT_COUNT), dtype=np.float32)
        self.action = np.zeros((self.memory_max), dtype=np.int32)
        self.reward = np.zeros((self.memory_max), dtype=np.float32)
        self.is_done = np.zeros((self.memory_max), dtype=np.bool)

    def store(self, state, action, reward, next_state, is_done):
        self.state[self.mem_step] = state
        self.next_state[self.mem_step] = next_state
        self.action[self.mem_step] = action
        self.reward[self.mem_step] = reward
        self.is_done[self.mem_step] = is_done
        self.update_mem_step()

    def update_mem_step(self):
        self.mem_step += 1
        if self.current_memory_size < self.memory_max:
            self.current_memory_size += 1
        if not self.batch_available:
            if self.mem_step > self.batch_size:
                self.batch_available = True
        if self.mem_step == self.memory_max:
            self.mem_step = 0

    def batch_ready(self):
        return self.batch_available

    def get_batch(self):
        output_indexes = np.random.choice(self.current_memory_size, self.batch_size)
        output_state = self.state[output_indexes]
        output_next_state = self.next_state[output_indexes]
        output_action = self.action[output_indexes]
        output_reward = self.reward[output_indexes]
        output_is_done = self.is_done[output_indexes]

        return [output_state, output_next_state, output_action, output_reward, output_is_done]
