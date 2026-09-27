import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import game
from helpers import state_gen, board_gen

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
            q_values = self.valid_actions(state,self.forward(state).tolist())
        #print("State:", state)
        chosen = q_values.index(max(q_values))
        #print("Q-values:", q_values.cpu().detach().numpy())
        #print("Chosen:", chosen)
        #print(state)
        if random.random()<epsilon:
            return random.randint(0,3),0
        else:
            return int(chosen),1

    def valid_actions(self,current_state,q_values):
        current_state = current_state.tolist()
        masking_board = game.Board()
        neg_inf = float('-inf')
        valid_action = [neg_inf for _ in range(4)]
        actions = [masking_board.move_up,masking_board.move_left,masking_board.move_right,masking_board.move_down]
        for id,fn in enumerate(actions):
            masking_board.board_state = board_gen(current_state)
            fn()
            next_state = state_gen(masking_board.board_state)
            """
            test_state = board_gen(current_state)
            round_trip = state_gen(test_state)

            print(current_state)
            print(round_trip)
            print(current_state == round_trip)
            """
            if current_state != next_state:
                valid_action[id] = 1
        #print(type(valid_action), valid_action)
        masked_values = [q_values[i] if valid_action[i] == 1 else valid_action[i] for i in range(4)]
        return masked_values
    
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
            Q_target_max = torch.max(target_network.forward(next_state),dim = 1).values

        target_value = reward + (1-done)*gamma*Q_target_max

        q_values = self.forward(state)
        row_indices = torch.arange(q_values.size(0))
        predicted_value = q_values[row_indices,action]
        loss = F.smooth_l1_loss(predicted_value,target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()