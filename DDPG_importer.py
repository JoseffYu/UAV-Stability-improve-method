import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
from collections import deque

Transitions = collections.namedtuple('Tramsitions', ['states', 'actions', 'rewards', 'next_states', 'dones'])

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transitions(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transitions(*zip(*transitions))
        return batch

    def size(self):
        return len(self.memory)


# Actor Net
class Actor(nn.Module):
    
    def __init__(self, n_states, n_hiddens, n_actions, max_action):
        super(Actor, self).__init__()
        # The max value for action
        self.max_action_value = max_action
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)  
        x = x * self.max_action_value  # 缩放到 [-max_action, max_action]
        return x
    
    
# Critic Net
class Critic(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Critic, self).__init__() 
        
        print("n_states: ",n_states)
        print("n_actions: ",n_actions)
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x, a):
        # Combind states and actions togethet
        cat = torch.cat([x, a], dim=1)
        print("cat: ",cat)
        x = self.fc1(cat)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        print("x: ",x)
        return x
    
    
# DDPG main body
class DDPG():
    
    def __init__(self, states_size:int, actions_size:int, max_action:float, batch_size:int, sigma:float, actor_lr:float, critic_lr:float, tau:float, gamma:float, device:str):
        self.states_size = states_size
        self.actions_size = actions_size
        self.batch_size = batch_size
        self.max_action = max_action
        self.sigma = sigma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        # Actor Net
        self.actor = Actor(states_size, 256, actions_size, max_action).to(device)
        self.actor_target = Actor(states_size, 256, actions_size, max_action).to(device)
        # Critic Net
        self.critic = Critic(states_size, 256, actions_size).to(device)
        self.critic_target = Critic(states_size, 256, actions_size).to(device)
        # Optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # Loss Fuction
        self.loss = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).view(1,-1).to(self.device)
        action = self.actor(state).item()
        return action + np.random.normal(0, self.sigma, size=self.states_size)
    
    def update(self, actor_net, actor_target):
        for target_param, param in zip(actor_target.parameters(), actor_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        
        batch_states = torch.tensor(batch.states, dtype=torch.float32).to(self.device) # get states from memory
        batch_actions = torch.tensor(batch.actions, dtype=torch.float32).to(self.device) # get actions from memory
        batch_rewards = torch.tensor(batch.rewards, dtype=torch.float32).to(self.device) # get rewards from memory
        batch_next_states = torch.tensor(batch.next_states, dtype=torch.float32).to(self.device) # get next_states from memory
        batch_dones = torch.tensor(batch.dones, dtype=torch.float32).to(self.device) # get dones from memory
        
        target_actions = self.actor_target(batch_states)
        target_action_q_values = self.critic_target(batch_states, target_actions)
        target_q_values = batch_rewards + self.gamma * (1 - batch_dones) * target_action_q_values
        
        print("breakpoint")
        current_q_values = self.critic(batch_states, batch_actions)
        
        critic_loss = self.loss(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(batch_states, self.actor(batch_states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.update(self.actor, self.actor_target)
        self.update(self.critic, self.critic_target)
