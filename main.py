
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import copy

class actor_net(nn.Module):
    def __init__(self, state_n, action_n, hidden_n):
        super(actor_net, self).__init__()
        self.fc1 = nn.Linear(state_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, hidden_n)
        self.fc3 = nn.Linear(hidden_n, action_n)
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        probs = f.softmax(self.fc3(x), dim = 1)
        return probs
class critic_net(nn.Module):
    def __init__(self, state_n, hidden_n):
        super(critic_net, self).__init__()
        self.fc1 = nn.Linear(state_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, hidden_n)
        self.fc3 = nn.Linear(hidden_n, 1)
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
class buffer(object):
    def __init__(self, length):
        self.buffer_length = length
        self.buffer = deque(maxlen = self.buffer_length)
    def push(self, trans):
        self.buffer.append(trans)
    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def length(self):
        return len(self.buffer)
    
    
class buffer(object):
    def __init__(self, length):
        self.buffer_length = length
        self.buffer = deque(maxlen = self.buffer_length)
    def push(self, trans):
        self.buffer.append(trans)
    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def length(self):
        return len(self.buffer)    
    
class config():
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.train_eps = 200
        self.test_eps = 20
        self.max_step = 400
        self.eval_eps = 5
        self.eval_per_ep = 10
        self.gamma = 0.99
        self.lambda_ = 0.99
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.buffer_length = 100
        self.eps_clip = 0.2
        self.entropy_coef = 0.01
        self.batch_size = 100
        self.update_n = 4
        self.hidden_n = 256
        self.seed = 42
        self.device = 'cuda'
        
class PPO():
    def __init__(self, cfg):
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.actor = actor_net(cfg.state_n, cfg.action_n, cfg.hidden_n).to(self.device)
        self.critic = critic_net(cfg.state_n, cfg.hidden_n).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = buffer(cfg.buffer_length)
        self.update_n = cfg.update_n
        self.eps_clip = cfg.eps_clip
        self.entropy_coef = cfg.entropy_coef
        self.sample_count = 0
        self.lambda_ = cfg.lambda_
 
        self.batch_size = cfg.batch_size
 
    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()
 
    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
 
    def update(self):
        if self.memory.length() < self.batch_size:
            return
        states, actions, log_probs, rewards, terminated, truncated, next_states  = self.memory.sample()
        
        #Use GAE here
        with torch.no_grad():
            states = torch.tensor(np.array(states), device=self.device)
            actions = torch.tensor(np.array(actions), device=self.device)
            log_probs = torch.tensor(log_probs, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(terminated or truncated, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), device=self.device)
            
            
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            advantage = torch.zeros_like(deltas)
            gae = 0

            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.lambda_ * (1 - dones[t]) * gae
                advantage[t] = gae

            returns = advantage + values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5) 
 
        for _ in range(self.update_n):
            values = self.critic(states)
            advantage = returns - values.detach()
            probs = self.actor(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            critic_loss = (returns - values).pow(2).mean()
 
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
        self.memory.clear()
        
        
def get_env_agent(cfg):
    #env = gym.make(cfg.env_name)            # For Headless Training
    env = gym.make(cfg.env_name, render_mode="human")
    state_n = env.observation_space.shape[0] #Position, Velocity, stick theta, stick theta dot
    action_n = env.action_space.n            #car to left, car to right
    print('n state: ', state_n)
    print('n action', action_n)
    setattr(cfg, 'state_n', state_n)
    setattr(cfg, 'action_n', action_n)
    agent = PPO(cfg)
    return env, agent

def train(cfg, env, agent):
    print('train')
    rewards = []
    steps = []
    best_ep_reward = 0
    output_agent = None
    for ep_i in range(cfg.train_eps):
        ep_reward = 0
        ep_step = 0
        state, info = env.reset(seed = cfg.seed)
        for _ in range(cfg.max_step):
            ep_step += 1
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.memory.push((state, action, agent.log_probs, reward, terminated, truncated, next_state))
            state = next_state
            agent.update()
            ep_reward += reward
            if terminated or truncated:
                break
        if (ep_i + 1) % cfg.eval_per_ep == 0:
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                state, _ = env.reset()
                for _ in range(cfg.max_step):
                    action = agent.predict_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    state = next_state
                    eval_ep_reward += reward    
                    if terminated or truncated:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward / cfg.eval_eps
            if mean_eval_reward > best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print('train ep_i:%d/%d, rewards:%f, mean_eval_reward:%f, best_ep_reward:%f, update model'%(ep_i + 1, cfg.train_eps, ep_reward, mean_eval_reward, best_ep_reward))
            else:
                print('train ep_i:%d/%d, rewards:%f, mean_eval_reward:%f, best_ep_reward:%f'%(ep_i + 1, cfg.train_eps, ep_reward, mean_eval_reward, best_ep_reward))
        steps.append(ep_step)
        rewards.append(ep_reward)
    env.close()
    return output_agent, rewards


def test(cfg, env, agent):
    print('test')
    rewards = []
    steps = []
    env = gym.make(cfg.env_name, render_mode="human") 
    for ep_i in range(cfg.test_eps):
        ep_reward = 0
        ep_step = 0
        state, _ = env.reset()
        for _ in range(cfg.max_step):
            ep_step += 1
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print('test ep_i:%d, reward:%f'%(ep_i + 1, ep_reward))
    env.close()
    return rewards



def smooth(data, weight = 0.9):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == '__main__':
    cfg = config()
    env, agent = get_env_agent(cfg)
    better_agent, train_rewards = train(cfg, env, agent)
    plt.figure()
    plt.title('training rewards')
    plt.plot(train_rewards, label = 'train_rewards')
    plt.plot(smooth(train_rewards), label = 'train_smooth_rewards')
 
    test_rewards = test(cfg, env, better_agent)
    plt.figure()
    plt.title('testing rewards')
    plt.plot(test_rewards, label = 'test_rewards')
    plt.plot(smooth(test_rewards), label = 'test_smooth_ewards')
    plt.show()
