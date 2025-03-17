import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from DDPG_importer import DDPG, ReplayBuffer
from matplotlib.animation import FuncAnimation

# Main body
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]
device = 'cpu'

agent = DDPG(states_size=state_size,
             actions_size=action_size,
             max_action=max_action,
             batch_size=128,
             sigma=0.001,
             actor_lr=0.95,
             critic_lr=0.95,
             tau=0.001,
             gamma=0.001,
             device=device)  # Ensure the device is specified correctly
replay = ReplayBuffer(capacity=100000)

all_returns = []
mean_returns = []

# 设置绘图
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 1000)
ax.set_ylim(-200, 0)
ax.set_xlabel('Episode')
ax.set_ylabel('Return')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(range(len(all_returns)), all_returns)
    return line,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True)

for episode in range(300):
    state = env.reset()[0]
    done = False
    info = False
    episode_return = 0
    while not done:
        env.render()
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
    print("Episode:", episode)
    print("Episode: {}, Return: {}, Mean Return: {}".format(episode, episode_return, mean_return))

env.close()

# 显示动画
plt.show()
