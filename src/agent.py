import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # Replay Buffer Size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # Learning rate
UPDATE_EVERY = 4        # How often to update the network
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """ Interacts with and learns from the enviroment """

    def __init__(self, state_size, action_size, seed):
        """ Initialize an Agent object

        Params
        ===
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        random.seed(self.seed)

        # Q Networks
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed, DEVICE)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ Returns the possible actions on the given state according to the current policy

        Params
        ===
            state (array_like): current states
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())

        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ Update value parameters using batch of experience tuples

        Params
        ===
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current state
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """ Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parametes(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)
