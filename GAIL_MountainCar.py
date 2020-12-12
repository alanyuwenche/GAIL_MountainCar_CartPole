# MountainCar- GAN+PPO( 0- left, 2- right)
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
class Discriminator(nn.Module):
    def __init__(self, s_dim, N_action):
        super(Discriminator, self).__init__()
        self.s_dim = s_dim
        self.N_action = N_action
        self.fc1 = nn.Linear(self.s_dim + self.N_action, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, s_dim, N_action):
        super(ActorCritic, self).__init__()
        self.s_dim = s_dim
        self.N_action = N_action
        self.action_layer = nn.Sequential(
                nn.Linear(self.s_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, N_action),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(self.s_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                )
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), action_probs, log_prob
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        actInx = torch.argmax(action,dim=1)
        action_logprobs = dist.log_prob(actInx)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
class GAIL:
    def __init__(self, K_epo, eps_clip, s_dim, N_action, gamma=0.99):
        self.s_dim = s_dim
        self.N_action = N_action
        self.gamma = gamma
        self.K_epochs = K_epo
        self.eps_clip = eps_clip
        self.policy = ActorCritic(self.s_dim, self.N_action).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.policy_loss = nn.MSELoss()
        self.policy_old = ActorCritic(self.s_dim, self.N_action).to(device)
        #self.policy_L = 0.
        
        self.disc = Discriminator(self.s_dim, self.N_action)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=1e-3)
        self.disc_loss = nn.BCELoss()
        #self.disc_L = 0
        
        self.policy_old.load_state_dict(self.policy.state_dict())



    def int_to_tensor(self, action):
        action = int(action)
        temp = torch.zeros(1, self.N_action)
        temp[0, action] = 1
        return temp
    

    
    def update(self, memory, e_s1_list, e_a1_list):
        # train Discriminator
        s1_list = []
        a1_list = []
        for s, a in zip(memory.states, memory.actions):
            s1_list.append(s)
            a1_list.append(a)
        p_s1 = torch.from_numpy(s1_list[0]).float()
        p_s1 = torch.reshape(p_s1,(1,self.s_dim))
        p_a1 = self.int_to_tensor(a1_list[0])
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.from_numpy(s1_list[i]).float()
            temp_p_s1 = torch.reshape(temp_p_s1,(1,self.s_dim))
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            temp_p_a1 = self.int_to_tensor(a1_list[i])
            p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)
        
        e_s1 = torch.from_numpy(e_s1_list[0]).float()
        e_s1 = torch.reshape(e_s1,(1,self.s_dim))
        e_a1 = self.int_to_tensor(e_a1_list[0])
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.from_numpy(e_s1_list[i]).float()
            temp_e_s1 = torch.reshape(temp_e_s1,(1,self.s_dim))
            e_s1 = torch.cat([e_s1, temp_e_s1], dim=0)
            temp_e_a1 = self.int_to_tensor(e_a1_list[i])
            e_a1 = torch.cat([e_a1, temp_e_a1], dim=0)

        p1_label = torch.zeros(len(s1_list), 1)
        e1_label = torch.ones(len(e_s1_list), 1)
        
        e1_pred = self.disc(e_s1, e_a1)
        loss = self.disc_loss(e1_pred, e1_label)
        p1_pred = self.disc(p_s1, p_a1)
        loss = loss + self.disc_loss(p1_pred, p1_label)
        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        #self.disc_L = loss        
        #####################################################
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = p_s1.to(device).detach()
        old_actions = p_a1.to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_pp = -torch.min(surr1, surr2) + 0.5*self.policy_loss(state_values, rewards) - 0.01*dist_entropy#ori

            # take gradient step
            self.policy_optimizer.zero_grad()
            loss_pp.mean().backward()
            self.policy_optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())#updated sucessfully
        #self.policy_L = loss_pp.mean()
        
def sample(exp_states, exp_actions, batch):
    n = len(exp_states)
    indexes = np.random.randint(0, n, size=batch)
    state, action = [], []
    for i in indexes:
        s = exp_states[i]
        a = exp_actions[i]
        state.append(np.array(s, copy=False))
        action.append(np.array(a, copy=False))
    return state, action
    
#env = gym.make("MountainCar-v0")
env = gym.make("CartPole-v0")#OK!!!!
s_dim = env.observation_space.shape[0]#MountainCar-v0: 2
N_action = env.action_space.n#MountainCar-v0: 3
max_epi_iter = 20000
max_MC_iter = 210 
batch_size = 200
K_epochs = 4      # update PPO policy for K epochs
eps_clip = 0.2    # clip parameter for PPO
solved_reward = 190  # stop training if solved_reward > avg_reward-MountainCar-v0: -110
n_eval_episodes = 20

exp_s_list = list(np.loadtxt("CartPole-v0_expert_states.csv"))#CartPole-v0    
exp_a_list = list(np.loadtxt("CartPole-v0_expert_actions.csv"))#CartPole-v0
#exp_s_list = list(np.loadtxt("MountainCar-v0_expert_states.csv"))#MountainCar-v0    
#exp_a_list = list(np.loadtxt("MountainCar-v0_expert_actions.csv"))#MountainCar-v0
exp_a_list = [int(i)for i in exp_a_list] # Converting
# generative adversarial imitation learning from [exp_s_list, exp_a_list]
agent = GAIL(K_epochs, eps_clip, s_dim, N_action)
memory = Memory()

avgList = []
timestep = 0
for epi_iter in range(max_epi_iter):
    state = env.reset()
    acc_r = 0
    for MC_iter in range(max_MC_iter):
        timestep += 1
        # env.render()
        action1, pi_a1, log_prob1 = agent.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action1)
        memory.logprobs.append(log_prob1)
        state, reward, done, _ = env.step(action1)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        acc_r = acc_r + reward
        
        if timestep % batch_size == 0:
            exp_s_samp, exp_a_samp = sample(exp_s_list, exp_a_list, batch_size)
            agent.update(memory, exp_s_samp, exp_a_samp)
            #agent.update(memory, exp_s_list, exp_a_list)
            #print('agent.policy_L: ',agent.policy_L.detach())
            memory.clear_memory()

        if done:
            break
        
    print('Imitate by GAIL, Episode', epi_iter, 'reward', acc_r)
    if len(avgList) < n_eval_episodes:
        avgList.append(acc_r)
    else:
        avgList.append(acc_r)
        avgList.pop(0)
    avg_reward = sum(avgList)/len(avgList)
    #print("avg_reward: ",avg_reward)
    if avg_reward > solved_reward:
        print("########### Solved! ###########")
        #agent.save(directory, filename)
        break
    

    

