import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """
    A network which approximates the action (Q) values.
    Values for all actions are predicted simultaneously as a function of state.
    Action values are decomposed into the sum of state values (common across all states)
    and advantages (action-specific, mean-0 across all actions).
    """
    
    def __init__(self, state_size, action_size, num_frames, seed, hidden_sizes_list):
        """Initialize parameters and build model.
        
        Inputs:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            num_frames (int): Number of previous frames included in the state.
            seed (int): Random seed.
            hidden_sizes_list (list): The sizes of hidden layers in the Q network.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        model = OrderedDict([
            ['fc_1', nn.Linear(state_size * (num_frames + 1) + num_frames, hidden_sizes_list[0])],
            ['relu_1', nn.ReLU()]
        ])
        for i in range(1, len(hidden_sizes_list)):
            model['fc_{}'.format(i + 1)] = nn.Linear(hidden_sizes_list[i - 1], hidden_sizes_list[i])
            model['relu_{}'.format(i + 1)] = nn.ReLU()
        self.stage1_model = nn.Sequential(model)
        self.fc_advantage = nn.Linear(hidden_sizes_list[-1], action_size)
        self.fc_state = nn.Linear(hidden_sizes_list[-1], 1)

    def forward(self, state):
        """ Forward propagation through the network."""
        stage_1 = self.stage1_model(state)
        advantages = self.fc_advantage(stage_1)
        values = self.fc_state(stage_1)
        return advantages - advantages.mean() + values