import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import game
from helpers import state_gen, board_gen, valid_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self,learning_rate,weight_decay):
        super(Agent,self).__init__()
        self.input = nn.Linear(16,64)
        self.layer1 = nn.Linear(64,64)
        self.layer2 = nn.Linear(64,64)
        self.output = nn.Linear(64,4)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay = weight_decay)
    def forward(self,x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

    
    def choice(self,state,epsilon):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = valid_actions(state,self.forward(state).tolist())
        #print("State:", state)
        #print(q_values)
        chosen = q_values.index(max(q_values))
        #print("Q-values:", q_values.cpu().detach().numpy())
        #print("Chosen:", chosen)
        #print(state)
        if random.random()<epsilon:
            return random.randint(0,3),0
        else:
            return int(chosen),1


    
    def update(self,batch,target_network,gamma):
        state = torch.tensor([row[0] for row in batch]).to(device)
        action = torch.tensor([row[1] for row in batch]).to(device)
        next_state = torch.tensor([row[2] for row in batch]).to(device)
        reward = torch.tensor([row[3] for row in batch]).to(device)
        done = torch.tensor([row[4] for row in batch], dtype=torch.float32).to(device)
        '''print(
            "Shapes:",
            state.shape,
            action.shape,
            next_state.shape,
            reward.shape,
            done.shape
        )'''
        vmap_valid_actions = torch.vmap(valid_actions, in_dims=(0, 0))
        with torch.no_grad():
            raw_target_q = target_network(next_state) # Shape: (B, 4)
            # Enforce valid_actions on each sample in the batch
            masked_target_q_list = [
                valid_actions(ns, q_val) 
                for ns, q_val in zip(next_state, raw_target_q)
            ]
            
            # Convert list of 1D tensors/lists back into a 2D Tensor of shape (B, 4)
            target_q_values = torch.tensor(masked_target_q_list, device=next_state.device, dtype=next_state.dtype)
            #print(target_q_values.tolist())
            Q_target_max = torch.max(target_q_values,dim = 1).values

        target_value = reward + (1-done)*gamma*Q_target_max

        q_values = self.forward(state)
        row_indices = torch.arange(q_values.size(0))
        predicted_value = q_values[row_indices,action]
        loss = F.smooth_l1_loss(predicted_value,target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()