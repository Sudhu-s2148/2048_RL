import game
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()
        self.input = nn.Linear(16,64)
        self.layer1 = nn.Linear(64,64)
        self.layer2 = nn.Linear(64,64)
        self.output = nn.Linear(64,4)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.0003,weight_decay =1e-4)
    def forward(self,x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x
    def choice(self,q_values,epsilon):
        if random.random()<epsilon:
            return random.randint(0,3)
        else:
            return int(q_values.argmax())

    def update(self,batch,target_network):
        gamma = 0.9
        for i in range(len(batch)):
            state = batch[i][0]
            action = batch[i][1]
            next_state = batch[i][2]
            reward = batch[i][3]
            done = batch[i][4]

            Q_target_max = target_network.forward(next_state).max()
            if done:
                target_value = reward
            else:
                target_value = reward + gamma*Q_target_max

            predicted_value = self.forward(state)[action]
            loss = F.smooth_l1_loss(predicted_value,target_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()