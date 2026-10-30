import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import game
import math
from helpers import state_gen, board_gen, valid_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self,learning_rate,weight_decay):
        super(Agent,self).__init__()
        self.input = nn.Linear(16,64)
        self.layer1 = nn.Linear(64,64)
        self.layer2 = nn.Linear(64,64)
        self.Q_output = nn.Linear(64,4)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay = weight_decay)

    def forward(self,x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        Q = self.Q_output(x)

        return Q

    def choice(self,state,epsilon):
        state = torch.tensor(state, dtype=torch.float32).to(device)

        '''
        with torch.no_grad():
            q_values = valid_actions(state,self.forward(state).tolist())
        
        chosen = q_values.index(max(q_values))
        '''

        with torch.no_grad():
            q_values = self.forward(state)
        
        chosen = torch.argmax(q_values).item()

        #print("State:", state)
        #print("State:", state)
        #print(q_values)
        #print("valdity mask:",mask)
        #print("Q-values:", q_values.cpu().detach().numpy())
        #print("Chosen:", chosen)
        #print(state)
        
        if random.random() < epsilon:
            return random.randint(0,3), 0
        
        else:
            return int(chosen),1


    
    def update(self,batch,target_network,online_network,gamma):
        state = torch.tensor([row[0] for row in batch],dtype = torch.float32,device=device)
        action = torch.tensor([row[1] for row in batch],dtype = torch.long,device=device)
        next_state = torch.tensor([row[2] for row in batch],dtype = torch.float32,device=device)
        reward = torch.tensor([row[3] for row in batch],dtype = torch.float32,device=device)
        done = torch.tensor([row[4] for row in batch], dtype=torch.float32).to(device)

        '''print(
            "Shapes:",
            state.shape,
            action.shape,
            next_state.shape,
            reward.shape,
            done.shape
        )'''
        #vmap_valid_actions = torch.vmap(valid_actions, in_dims=(0, 0))
        with torch.no_grad():
            online_q_values = online_network(next_state) # Shape: (B, 4)
            chosen_action = torch.argmax(online_q_values,dim = 1)
            target_q_values = target_network(next_state)
            Q_target_max = target_q_values.gather(1, chosen_action.unsqueeze(1)).squeeze(1)
            Q_target_max = torch.where(done.bool(),torch.zeros_like(Q_target_max),Q_target_max)

        target_q_value = reward + (1-done)*gamma*Q_target_max
        q_values = self.forward(state)
        row_indices = torch.arange(q_values.size(0))
        predicted_q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)


        
        q_loss = F.smooth_l1_loss(predicted_q_value,target_q_value)

        loss = q_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()