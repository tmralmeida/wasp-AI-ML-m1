import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np
from collections import namedtuple
import scipy.signal


def combined_shape(length, shape=None):
    """Returns the correct shape based on the type of oobservation and actions spaces
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


Experience = namedtuple("Experience", (
    "state, action, reward"
))


def mlp(sizes, activation = nn.Tanh, last_activation = nn.Identity):
    """ Function that returns a multi layer perceptron pytorch object. It's the policy network
    Args:
        sizes (iterator): hidden sizes 
        activation (class 'torch.nn.modules.activation): activation of the hidden layers. Defaults to nn.Tanh.
        last_activation (class 'torch.nn.modules.activation): [activation of the last layer]. Defaults to nn.Identity.
    """
    s = list(zip(sizes[::1], sizes[1::])) # pairs together the respectiive sizes of each FC layer
    layers = [] 
    for shape in s[:-1]:
        layers.extend([nn.Linear(shape[0], shape[1]), activation()])
    layers.extend([nn.Linear(s[-1][0], s[-1][1]), last_activation()])
    return nn.Sequential(*layers)



# Replay buffer
class ReplayMemory():
    """Replay buffer object
    
    Attributes:
        capacity (int): max number of experiences that the buffer can store
        gamma (float): discount factor
        lam (float): baseline parameter
        push_count (int): current index in the buffer
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, adv_buf (np.array): buffers
        
    Methods:
        reset_count(): reset the counter
        fill_buffer(obs,act,rew,v,logp): store experiences in the buffer
        push(experience, v): appends an experience to the buffer and the respective value and logp 
        get(): returns the full buffer
        
    """
    def __init__(self, capacity, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.obs_buf = np.zeros(combined_shape(capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.push_count, self.path_start_idx = 0, 0
        
    def reset_count(self):
        self.push_count = 0
        
    def fill_buffer(self, obs, act, rew, v):
        self.obs_buf[self.push_count] = obs.cpu() if isinstance(obs, torch.Tensor) else obs
        self.act_buf[self.push_count] = act
        self.rew_buf[self.push_count] = rew
        self.val_buf[self.push_count] = v
        
        
    def push(self, experience, v):
        obs, act, rew = experience.state, experience.action, experience.reward
        if self.push_count >= self.capacity: self.reset_count()
        self.fill_buffer(obs, act, rew, v)    
        self.push_count += 1   


    def finish_path(self, last_val = 0):
        """Based on: https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/vpg/core.py#L29
        """
        path_slice = slice(self.path_start_idx, self.push_count)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.push_count
        
    def get(self):
        #adv_mean, adv_std = np.mean(np.array(self.adv_buf)), np.std(np.array(self.adv_buf))
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.rew_buf,
                    adv=self.adv_buf)
        for k, v in data.items():
            {k: torch.as_tensor(v, dtype=torch.float32)}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    

# Policy and value functions
class MLPCategoricalPolicy(nn.Module):
    """ Handles the full operation of an policy of discrete actions
    Attributes:
        policy_net (nn.Sequential): NN that maps observations into the actions' logits
    Methods:
        dist(obs): computes the logits of the observation and returns a categorical dist based on that
        log_prob_from_dist(pi, act): computes the log likelihood of an observation in a certain policy dist
        forward(obs, optional act): handles the full process: get a dist and computes the log likelihood 
    """
    def __init__(self, sizes, activation = nn.Tanh, last_activation = nn.Identity, device = torch.device("cpu")):
        super().__init__()
        self.policy_net = mlp(sizes, activation, last_activation).to(device)
        
    def dist(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits = logits)
    
    def log_prob_from_dist(self, pi, act):
        return pi.log_prob(act)
    
    def forward(self, obs, act = None):
        pi = self.dist(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a
    
    
    
class PolicyValue(nn.Module):
    """ Highlevel handler of the Policy and Value functions. Although several res define REINFORCE as an AC, I don't agree
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation = nn.Tanh, last_activation = nn.Identity, device = torch.device("cpu")):
        super().__init__()
        
        
        dims = [np.prod(observation_space.shape)] + list(hidden_sizes)
        
            
        # Value Function
        self.v_pi = mlp(dims + [1], activation, last_activation).to(device)
        
        self.pi = MLPCategoricalPolicy(dims + [action_space.shape[0]], activation, last_activation, device)
            
                
    def step(self, obs):
        obs = obs.flatten()
        with torch.no_grad():
            pi = self.pi.dist(obs)
            a = pi.sample()
            v = self.v_pi(obs).squeeze()
        return a.cpu().numpy(), v.cpu().numpy()