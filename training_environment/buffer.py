from collections import deque
import random
class exp_buffer():
    def __init__(self,max_size):
        self.buffer = deque()
        self.max_length = max_size
    def append(self,exp):
        if len(self.buffer) == self.max_length:
            self.buffer.popleft()
        self.buffer.append(exp)
    def sample(self,batch_size):
        training_batch = random.sample(self.buffer,batch_size)
        return training_batch
    def __len__(self):
            return len(self.buffer)