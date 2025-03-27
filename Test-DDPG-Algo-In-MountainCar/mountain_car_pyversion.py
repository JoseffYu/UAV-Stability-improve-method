
import wandb
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from DDPG_importer import DDPG, ReplayBuffer
from matplotlib.animation import FuncAnimation

wandb.init(
    project="mountain_car_algo_test",  # 项目名称
    config={                     # 记录超参数
        "init_sigma":0.8,
        "actor_lr":1e-5,
        "critic_lr":1e-4,
        "tau":0.001,
        "gamma":0.99,
        "batch_size":256}
)

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
             batch_size=256,
             init_sigma=0.9,
             final_sigma=0.5,
             actor_lr=1e-5,
             critic_lr=1e-4,
             tau=0.001,
             gamma=0.99,
             device=device)  # Ensure the device is specified correctly
replay = ReplayBuffer(capacity=100000)

all_returns = []
mean_returns = []

for episode in range(200):
    state = env.reset()[0]
    done = False
    info = False
    episode_return = 0
    step_count = 0
    while not done:
        step_count += 1
        if step_count > 2000:
            print("Break Training")
            break
        action = agent.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        agent.steps += 1
        replay.push(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        if replay.size() > 10000:
            agent.train(replay)
    all_returns.append(episode_return)
    mean_return = np.mean(all_returns)
    mean_returns.append(mean_return)
    agent.episodes += 1
    wandb.log({
        "Episode Reward": episode_return,
        "Mean Reward": mean_return,
        "Episode": episode
    })
    print("Episode: {}, Return: {}, Mean Return: {}".format(episode, episode_return, mean_return))

env.close()
wandb.finish()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(all_returns) + 1), all_returns, label='Return')
plt.plot(range(1, len(mean_returns) + 1), mean_returns, label='Mean Return', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Return vs Episode')
plt.legend()
plt.grid(True)
plt.show()