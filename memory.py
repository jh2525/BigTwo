
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, discount_factor: float, gae_factor: float, gae: bool, gae_target: bool):
        self.clean()
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.gae = gae
        self.gae_target = gae_target
    
    def append(self, state: torch.Tensor, log_prob: float, state_value: float, reward: float, action: int):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.actions.append(action)
    
    def clean(self):
        self.states = []
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.actions = []
    
    def get_tensor(self):
        rewards = torch.Tensor(self.rewards)

        t_max = len(rewards) - 1

        advantages = torch.zeros((t_max + 1, ))
        discounted_retuns = torch.zeros((t_max + 1, ))
        discounted_retuns[t_max] = rewards[t_max]
        for t in reversed(range(t_max)):
            discounted_retuns[t] = discounted_retuns[t+1] * self.discount_factor
        V = torch.Tensor(self.state_values)

        if self.gae:
            advantages[t_max] = rewards[t_max] - V[t_max]
            for t in reversed(range(t_max)):
                delta_t = rewards[t] + self.discount_factor * self.state_values[t + 1] - self.state_values[t]
                advantages[t] = self.gae_factor * self.discount_factor * advantages[t] + delta_t
        else:
            advantages =  discounted_retuns - V

        if self.gae_target:
            target_action_values = advantages + V
        else:
            target_action_values = discounted_retuns
            
        max_cases = 1
        states = self.states
        for state in states:
            n_cases = state.size(0)
            max_cases = n_cases if n_cases > max_cases else max_cases
            
        masking = torch.zeros((len(states), max_cases))
        
        for i in range(len(states)):
            n_cases = states[i].size(0)
            masking[i, :n_cases] = 1.0
            
        state_tensors = torch.zeros((len(states), max_cases) + states[0].shape[1:])

        for i in range(len(states)):
            state_tensors[i, :states[i].size(0)] = states[i]
        
        return {
            'target_action_values' : target_action_values,
            'advantages' : advantages,
            'log_prob' : torch.Tensor(self.log_probs),
            'states' : state_tensors,
            'actions' : torch.Tensor(self.actions),
            'masking' : masking
        }

class ASPPOMemory:
    def __init__(self, discount_factor: float, gae_factor: float, gae: bool, gae_target: bool):
        self.clean()
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.gae = gae
        self.gae_target = gae_target
    
    def append(self, action_state: torch.Tensor, reward: float, state_value: float, log_prob: float, rho: float, max_logit: float):
        self.action_states.append(action_state)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.rhos.append(rho)
        self.max_logit.append(max_logit)

    def clean(self):
        self.action_states = []
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.actions = []
        self.rhos = []
        self.max_logit = []
    
    def get_tensor(self):
        rewards = torch.Tensor(self.rewards)

        t_max = len(rewards) - 1

        advantages = torch.zeros((t_max + 1, ))
        discounted_retuns = torch.zeros((t_max + 1, ))
        discounted_retuns[t_max] = rewards[t_max]
        for t in reversed(range(t_max)):
            discounted_retuns[t] = discounted_retuns[t+1] * self.discount_factor
        V = torch.Tensor(self.state_values)

        if self.gae:
            advantages[t_max] = rewards[t_max] - V[t_max]
            for t in reversed(range(t_max)):
                delta_t = rewards[t] + self.discount_factor * self.state_values[t + 1] - self.state_values[t]
                advantages[t] = self.gae_factor * self.discount_factor * advantages[t] + delta_t
        else:
            advantages =  discounted_retuns - V

        if self.gae_target:
            target_action_values = advantages + V
        else:
            target_action_values = discounted_retuns
        
        return {
            'target_action_values' : target_action_values,
            'advantages' : advantages,
            'log_prob' : torch.Tensor(self.log_probs),
            'action_states' : torch.concat(self.action_states, dim=0),
            'rhos' : torch.stack(self.rhos, dim=0),
            'max_logits': torch.stack(self.max_logit, dim=0)
        }
    
class RPOMemory:
    def __init__(self, discount_factor: float, gae_factor: float, gae: bool, gae_target: bool):
        self.clean()
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.gae = gae
        self.gae_target = gae_target
    
    def append(self, state: torch.Tensor, action_state:torch.Tensor, log_prob: float, state_value: float, reward: float, action: int):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.actions.append(action)
        self.action_states.append(action_state)
    
    def clean(self):
        self.states = []
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.action_states = []
    
    def get_tensor(self):
        rewards = torch.Tensor(self.rewards)

        t_max = len(rewards) - 1

        advantages = torch.zeros((t_max + 1, ))
        discounted_retuns = torch.zeros((t_max + 1, ))
        discounted_retuns[t_max] = rewards[t_max]
        for t in reversed(range(t_max)):
            discounted_retuns[t] = discounted_retuns[t+1] * self.discount_factor
        V = torch.Tensor(self.state_values)

        if self.gae:
            advantages[t_max] = rewards[t_max] - V[t_max]
            for t in reversed(range(t_max)):
                delta_t = rewards[t] + self.discount_factor * self.state_values[t + 1] - self.state_values[t]
                advantages[t] = self.gae_factor * self.discount_factor * advantages[t] + delta_t
        else:
            advantages =  discounted_retuns - V

        if self.gae_target:
            target_action_values = advantages + V
        else:
            target_action_values = discounted_retuns
            
        max_cases = 1
        action_states = self.action_states
        for action_state in action_states:
            n_cases = action_state.size(1)
            max_cases = n_cases if n_cases > max_cases else max_cases
            
        masking = torch.zeros((len(action_states), max_cases))

        for i in range(len(action_states)):
            n_cases = action_states[i].size(1)
            masking[i, :n_cases] = 1.0
            
        action_state_tensors = torch.zeros((len(action_states), max_cases, action_states[0].size(2)))

        for i in range(len(action_states)):
            action_state_tensors[i, :action_states[i].size(1)] = action_states[i]
        
        return {
            'target_action_values' : target_action_values,
            'advantages' : advantages,
            'log_prob' : torch.Tensor(self.log_probs),
            'states' : torch.concat(self.states, dim=0),
            'action_states' : action_state_tensors,
            'actions' : torch.Tensor(self.actions),
            'masking' : masking,
            'rewards' : torch.Tensor(self.rewards)
        }

class NoneMemory():
    def clean(self):
        self.rewards = [0]
        return
    def append(self, *args):
        return
    