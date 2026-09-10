import game
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy

class Agent(nn.Module):
    def __init__(self,learning_rate,weight_decay):
        super(Agent,self).__init__()
        self.input = nn.Linear(16,64)
        self.layer1 = nn.Linear(64,64)
        self.layer2 = nn.Linear(64,64)
        self.output = nn.Linear(64,4)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay = weight_decay)
    def forward(self,x):
        x = torch.tensor(x)
        x = F.relu(self.input(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x
    def choice(self,state,epsilon):
        q_values = self.forward(state)
        print("State:", state)
        print("Q-values:", q_values.detach().numpy())
        print("Chosen:", torch.argmax(q_values).item())
        if random.random()<epsilon:
            return random.randint(0,3)
        else:
            return int(q_values.argmax())

    def update(self,batch,target_network,gamma):
        state = torch.tensor([row[0] for row in batch])
        action = torch.tensor([row[1] for row in batch])
        next_state = torch.tensor([row[2] for row in batch])
        reward = torch.tensor([row[3] for row in batch])
        done = torch.tensor([row[4] for row in batch], dtype=torch.float32)
        print(
            "Shapes:",
            state.shape,
            action.shape,
            next_state.shape,
            reward.shape,
            done.shape
        )
        with torch.no_grad():
            Q_target_max = torch.max(target_network.forward(next_state),dim = 1).values

        target_value = reward + (1-done)*gamma*Q_target_max

        q_values = self.forward(state)
        row_indices = torch.arange(q_values.size(0))
        predicted_value = q_values[row_indices,action]
        loss = F.smooth_l1_loss(predicted_value,target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()