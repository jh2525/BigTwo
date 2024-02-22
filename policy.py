import torch
import numpy as np
from typing import Tuple, List
from torch.distributions.categorical import Categorical
import random

log_prob = float
state_value = float
action = int
rho = float 
max_logit = float


class Policy:
    def choose_action(self, observation: torch.Tensor) -> Tuple[log_prob, state_value, action]:
        pass

class RandomPolicy(Policy):
    def choose_action(self, observation: torch.Tensor) -> Tuple[log_prob, state_value, action]:
        if observation.size(0) == 1:
            return np.log(1/observation.size(0)), 0.0, 0
        
        return np.log(1/observation.size(0)), 0.0, np.random.randint(1, observation.size(0))

class ActorCriticPolicy(Policy):
    def __init__(self, model):
        self.model: torch.Module = model
    
    def choose_action(self, observation: torch.Tensor) -> Tuple[log_prob, state_value, action]:
        
        self.model.eval()

        
        with torch.no_grad():
            critic, logits = self.model(observation)
            
        critic, logits = critic.detach().flatten(), logits.detach().flatten()
        dist = Categorical(logits = logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.item()

        probs = torch.softmax(logits, dim=0)
        state_value = (probs * critic).sum()
        
        return log_prob.item(), state_value.item(), action


class ActorGreedyPolicy2(Policy):
    def __init__(self, model, greedy_p):
        self.model: torch.Module = model
        self.greedy_p: float = greedy_p
    
    def choose_action(self, observation: torch.Tensor) -> Tuple[log_prob, state_value, action, rho, max_logit]:
        
        self.model.eval()

        
        with torch.no_grad():
            action_values, logits = self.model(observation)
            
        action_values, logits = action_values.detach().flatten(), logits.detach().flatten()
        dist = Categorical(logits = logits)
        if random.random() < self.greedy_p:
            action = action_values.argmax()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.item()
        
        
        probs = torch.softmax(logits, dim=0)
        state_value = (probs * action_values).sum()
        max_logit = logits.max()
        rho = torch.exp(logits - max_logit).sum() - torch.exp(logits[action] - max_logit)
        
        return log_prob.item(), state_value.item(), action, rho, max_logit

class ActorGreedyPolicy(Policy):
    def __init__(self, model, greedy_p):
        self.model: torch.Module = model
        self.greedy_p: float = greedy_p
    
    def choose_action(self, observation: torch.Tensor) -> Tuple[log_prob, state_value, action]:
        
        self.model.eval()

        
        with torch.no_grad():
            action_values, logits = self.model(observation)

            
        action_values, logits = action_values.detach().flatten(), logits.detach().flatten()
       
        dist = Categorical(logits = logits)
        if random.random() < self.greedy_p:
            action = action_values.argmax()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.item()
        
        
        probs = torch.softmax(logits, dim=0)
        state_value = (probs * action_values).sum()
        
        return log_prob.item(), state_value.item(), action
