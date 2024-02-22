from typing import Optional, Tuple, List, Union
import numpy as np
from itertools import product
import torch
import random


players = 4
n_suit = 4 #spade, heart, clover, dia

card_coordi = Tuple[int, int]

suit_to_int = {'spade' : 3, 'heart' : 2, 'clover' : 1, 'diamond' : 0}

max_num = 13 #2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
num_to_int = {'2' : 0,
              '3' : 1,
              '4' : 2,
              '5' : 3,
              '6' : 4,
              '7' : 5,
              '8' : 6,
              '9' : 7,
              '10' : 8,
              'J' : 9,
              'Q' : 10,
              'K' : 11,
              'A' : 12}



sn_index = lambda k, n : (suit_to_int[k], num_to_int[n])

start_card_n = 13*4 // players

round_memory = 3


ranking_rewards = [
    1.0,
    0.5,
    -0.5,
    -1.0
    
]

