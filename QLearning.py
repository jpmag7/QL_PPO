import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class QNetwork(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes, hidden_activation=nn.LeakyReLU(), out_activation=nn.Softmax()):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential()
        prev_size = in_size
        for i, size in enumerate(hidden_sizes):
            self.layers.add_module(f"fc{i}", nn.Linear(prev_size, size))
            self.layers.add_module(f"act{i}", hidden_activation)
            prev_size = size
        self.layers.add_module("fc_out", nn.Linear(prev_size, out_size))
        self.layers.add_module("act_out", out_activation)

    def forward(self, state):
        x = state.float()
        for layer in self.layers:
            x = layer(x)
        return x

class QLearningAgent:
    def __init__(self, num_states, num_actions, hidden_sizes=[8, 8], learning_rate=0.01, discount_factor=0.9, exploration_rate=0, hidden_activation=nn.LeakyReLU(), out_activation=nn.Softmax()):
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.Q = QNetwork(num_states, num_actions, hidden_sizes, hidden_activation=hidden_activation, out_activation=out_activation)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        if np.random.uniform() < self.exploration_rate:
            # Choose a random action
            action = np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.Tensor(state))
                action = torch.argmax(Q_values).item()
        return action
    
    def update(self, episode):
        T = len(episode)
        states = torch.Tensor([step[0] for step in episode])
        actions = torch.LongTensor([step[1] for step in episode])
        rewards = torch.Tensor([step[2] for step in episode])
        next_states = torch.Tensor([step[3] for step in episode])
        Q_values = self.Q(states).gather(1, actions.unsqueeze(1))
        next_Q_values = self.Q(next_states).max(1)[0].detach()
        targets = rewards + self.discount_factor * next_Q_values
        loss = nn.functional.mse_loss(Q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


################################### EXAMPLE ############################################

def run_example():
    """
        A simple QLearning Agent environment that as to move left, right or maintain
        position on a 1D number line to follow a given random target in 20 steps
    """
    import random
    # Define a simple environment with 2 states and 3 actions
    num_states = 2
    num_actions = 3

    # Define a reward function that gives a reward based on current state and action applied to that state
    def reward_function(action, state, target):
        action = action - 1
        if abs(target - (state[0] + action)) < abs(target - state[0]):
            return 0.5
        elif state[0] + action == target:
            return 1
        else:
            return -0.5

    # Create a Q-learning agent
    agent = QLearningAgent(num_states, num_actions, hidden_sizes=[8, 8], learning_rate=0.01, discount_factor=0.9, exploration_rate=0.1)

    # Train the agent for 5000 episodes
    for i in range(1, 10001):
        target = random.randint(0, 20)
        state = [random.randint(0, 20), target]
        episode = []
        j = 1
        while j <= 20:
            j = j + 1
            action = agent.select_action(state)
            next_state = [state[0] + (action - 1), target]
            reward = reward_function(action, state, target)
            if j % 20 == 0 and i % 1000 == 0:
                print("\n############")
                print("State:", state)
                print("Action:", action - 1)
                print("Next State:", next_state)
                print("Reward:", reward)
            episode.append((state, action, reward, next_state))
            state = next_state
        agent.update(episode)
    
    # Test the agent
    agent.exploration_rate = 0
    for _ in range(10):
        target = random.randint(0, 20)
        state = [random.randint(0, 20), target]
        for _ in range(20):
            action = agent.select_action(state)
            next_state = [state[0] + (action - 1), target]
            reward = reward_function(action, state, target)
            print("State:", state, "Action:", action, "Reward:", reward)
            state = next_state

run_example()