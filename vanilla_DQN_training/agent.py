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

    
    def choice(self, state, epsilon):
        # valid_actions(state) returns a 4-element mask in action order:
        # [up, left, right, down].
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = self.forward(state_tensor)
            validity = torch.as_tensor(
                valid_actions(state_tensor.tolist()),
                dtype=torch.bool,
                device=q_values.device
            )

            if validity.any():
                masked_q_values = q_values.masked_fill(~validity, -torch.inf)
                valid_indices = torch.where(validity)[0]

                if random.random() < epsilon:
                    return int(random.choice(valid_indices.tolist())), 0

                return int(torch.argmax(masked_q_values).item()), 1

            # No legal moves (normally the game should already be over).
            return 0, 1


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
        with torch.no_grad():
            raw_target_q = target_network(next_state)  # Shape: (B, 4)
            masked_target_q_list = []

            for ns, q_values in zip(next_state, raw_target_q):
                validity = torch.as_tensor(
                    valid_actions(ns.tolist()),
                    dtype=torch.bool,
                    device=q_values.device
                )

                if validity.any():
                    masked_q = q_values.masked_fill(~validity, -torch.inf)
                else:
                    # No legal actions: use a zero bootstrap value to avoid NaNs.
                    masked_q = torch.zeros_like(q_values)

                masked_target_q_list.append(masked_q)

            target_q_values = torch.stack(masked_target_q_list)
            Q_target_max = torch.max(target_q_values, dim=1).values

        target_value = reward + (1-done)*gamma*Q_target_max

        q_values = self.forward(state)
        row_indices = torch.arange(q_values.size(0))
        predicted_value = q_values[row_indices,action]
        loss = F.smooth_l1_loss(predicted_value,target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()