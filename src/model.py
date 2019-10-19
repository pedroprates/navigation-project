import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ The agent's policy model """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Initialize the parameters and build the model
        Params
        ===
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QNetwork, self).__init__()

        self.seed = seed
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """ Perform the forward propagation of the neural network """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
 
        return self.fc3(x)
