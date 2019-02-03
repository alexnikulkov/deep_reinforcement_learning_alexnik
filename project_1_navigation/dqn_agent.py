import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weighted_mse_loss(pred, target, weights):
    return torch.sum(weights * (pred - target) ** 2)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Inputs:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.weights = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Inputs:
            state: Current state
            action: Action taken in the current state.
            reward: Reward received at the current step.
            next_state: A state into which the transition occurs.
            done: An indicator of the transition learding into a terminal state.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.weights.append(10)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = random.choices(range(len(self.memory)), weights=self.weights, k=self.batch_size)
        experiences = [self.memory[i] for i in indices]
        weights = [self.weights[i] for i in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_weights(self, indices, weights):
        assert len(indices) == len(weights)
        for i,w in zip(indices, weights):
            self.weights[i] = w


class Agent():
    """ An RL agent which can interact with an environment and learn from replayed experiences.
    """
    
    def __init__(self, state_size, action_size, seed, hidden_sizes_list, num_frames=2, num_iters_learn=1, update_every=4,\
                 batch_size=128, gamma=0.99, buffer_size=int(1e6), update_target_network_every=1, lr0=1e-3):
        """ Initialize an Agent object.
        
        Inputs:
            state_size (int): The dimensionality of the state space.
            action_size (int): Number of possible actions an agent can take.
            seed (int): Randomization seed.
            hidden_sizes_list (list): The sizes of hidden layers in the Q network.
            num_frames (int, optional): How many previous frames are part of the state (default 2).
            num_iters_learn (int, optional): Number of iterations to take at each step towards the targets.
            update_every (int, optional): How often themain network is updated (default 4).
            batch_size (int, optional): Batch size for each upate (default 128).
            gamma (float, optional): Temporal discount coefficient (default 0.99).
            buffer_size (int, optional): Maximum capacity of the replay buffer (default 1e5).
            update_target_network_every (int, optional): How often to update the target network (default 200).
            lr0 (float, optional): Initial learning rate (default 1e-3).
        """
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed
        self.num_frames = num_frames
        self.num_iters_learn = num_iters_learn
        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr0 = lr0
        self.update_target_network_every = update_target_network_every
        self.QNetwork_main = QNetwork(state_size, action_size, num_frames, seed, hidden_sizes_list).to(device)
        self.QNetwork_target = QNetwork(state_size, action_size, num_frames, seed + 1, hidden_sizes_list).to(device)
        self.optimizer_main = optim.Adam(self.QNetwork_main.parameters(), lr=lr0)
        self.optimizer_target = optim.Adam(self.QNetwork_target.parameters(), lr=lr0)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every "update_every" steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):       
        """Process a single state change.
        Periodically learn (update netork) if enough experiences are available in the replay buffer.
        
        Inputs:
            state: Current state
            action: Action taken in the current state.
            reward: Reward received at the current step.
            next_state: A state into which the transition occurs.
            done: An indicator of the transition learding into a terminal state.
        """
        self.t_step += 1
        if self.t_step % self.update_every == 0: # Learn every "update_every" time steps.
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Inputs:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.QNetwork_main.eval()
        with torch.no_grad():
            action_values = self.QNetwork_main(state)
        self.QNetwork_main.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
                
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Inputs:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences
        probs = np.array(weights) / sum(weights)
        is_weights = np.power(1.0 / (len(self.memory) * probs), 0.5)
        is_weights = is_weights / max(is_weights)

        with torch.no_grad():
            best_next_actions = self.QNetwork_main(next_states).argmax(dim=1)
            targets = rewards + gamma * self.QNetwork_target(next_states).gather(1, best_next_actions.unsqueeze(1)) * (1 - dones)
        for _ in range(self.num_iters_learn):
            self.optimizer_main.zero_grad()
            predictions = self.QNetwork_main(states).gather(1, actions)
            errors = (predictions - targets).squeeze()
            loss = weighted_mse_loss(predictions, targets, torch.from_numpy(is_weights).float())
            loss.backward()
            self.optimizer_main.step()
        self.memory.update_weights(indices, np.power(0.05 + errors.abs().data.numpy(), 0.5))
        
        if self.t_step % self.update_target_network_every == 0:
            self.soft_update(self.QNetwork_main, self.QNetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
