import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from DDPG_importer import DDPG, ReplayBuffer
from matplotlib.animation import FuncAnimation

# Main body
print(torch.cuda.is_available())  # 应输出True
env = gym.make("MountainCarContinuous-v0", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = DDPG(states_size=state_size,
             actions_size=action_size,
             max_action=max_action,
             batch_size=128,
             sigma=0.01,
             actor_lr=3e-3,
             critic_lr=3e-3,
             tau=0.001,
             gamma=0.9,
             device=device)  # Ensure the device is specified correctly
replay = ReplayBuffer(capacity=100000)

all_returns = []
mean_returns = []

for episode in range(500):
    state = env.reset()[0]
    done = False
    info = False
    episode_return = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        replay.push(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        if replay.size() > 500:
            agent.train(replay)
    all_returns.append(episode_return)
    mean_return = np.mean(all_returns)
    mean_returns.append(mean_return)
    print("Episode: {}, Return: {}, Mean Return: {}".format(episode, episode_return, mean_return))

env.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(all_returns) + 1), all_returns, label='Return')
plt.plot(range(1, len(mean_returns) + 1), mean_returns, label='Mean Return', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Return vs Episode')
plt.legend()
plt.grid(True)
plt.show()