import torch
from Big2.hand import Hand
from Big2.handranking import *
import torch.nn as nn

class Observer():
    def get_observation(self, logs, player_hand: Hand) -> torch.Tensor:
        return 
    
class DefaultObserver(Observer):
    
    def get_observation(self, logs, player_hand: Hand) -> torch.Tensor:
        
        
        pass_count = 0
        for i in range(3):
            if logs[-(i+1)]['pass']:
                pass_count += 1
            else:
                break
        
        strongest_ranking = logs[-1]['strongest_ranking'] if pass_count < 3 else HandRanking.NONE
        #all discard target card 
        target = torch.stack([ranking.card_tensor 
                              for ranking in player_hand.get_all_hand_rankings_stronger_than(strongest_ranking)], 
                              dim=0) #(n_action, 1, 4, 13)
        
        n_action = target.size(0)

        #hand associated with discards
        remain_hand = player_hand.card_tensor.expand_as(target) - target #(n_action, 1, 4, 13)
        #largest suit cards
        largest_suit_cards = strongest_ranking.card_tensor.expand_as(target) #(n_action, 1, 4, 13)
        #the number of remain hand
        n_cards = torch.zeros((1, 4, 13))
        for i in range(3):
            n_cards[:, -(i+1), int(logs[-(i+1)]['card_tensor'].sum().item() - 1)] = 1.0
        n_cards = n_cards.expand_as(target)
        n_cards = torch.index_put(n_cards, 
            (torch.arange(n_cards.size(0)).view(-1, 1), 
            torch.zeros((n_cards.size(0), 1), dtype = torch.long),
            torch.zeros((n_cards.size(0), 1), dtype = torch.long), 
            remain_hand.sum(dim=[1,2,3]).long().view(-1, 1)-1),
            values = torch.Tensor([1.0]))

        #all discard cards
        all_discarded_cards = torch.stack([log['discard'] for log in logs], dim=0).sum(dim=0).expand_as(target)

        observation = torch.concat([remain_hand, target, largest_suit_cards, n_cards, all_discarded_cards], dim=1).view(n_action, -1, 13)

        #pass count
        pass_count_tensor = torch.zeros((n_action, 1, 13))
        pass_count_tensor[:, 0, pass_count] = 1

        observation = torch.concat([observation, pass_count_tensor], dim=1)
        
        return observation
    

class RPOObserver(Observer):
    
    def get_observation(self, logs, player_hand: Hand) -> torch.Tensor:
        
        
        pass_count = 0
        for i in range(3):
            if logs[-(i+1)]['pass']:
                pass_count += 1
            else:
                break
        
        strongest_ranking = logs[-1]['strongest_ranking'] if pass_count < 3 else HandRanking.NONE
        #all discard target card 
        target = torch.concat([ranking.card_tensor.view(1, -1) 
                              for ranking in player_hand.get_all_hand_rankings_stronger_than(strongest_ranking)], 
                              dim=0) #(n_action, 1, 4, 13)
        
        player_card_tensor = player_hand.card_tensor.view(1, -1)
        n_action = target.size(0)

        
        #largest suit cards
        largest_suit_cards = strongest_ranking.card_tensor.view(1, -1) #(n_action, 1, 4, 13)
        #the number of remain hand
        n_cards = torch.zeros((1, 4, 13))
        for i in range(4):
            n_cards[:, -i, int(logs[-i]['card_tensor'].sum().item() - 1)] = 1.0

        n_cards = n_cards.view(1, -1)
        
        #all discard cards
        all_discarded_cards = torch.stack([log['discard'] for log in logs], dim=0).sum(dim=0).view(1, -1)

        #pass count
        pass_count_tensor = torch.zeros((1, 4))
        pass_count_tensor[:, pass_count] = 1

        
        
        return torch.concat([player_card_tensor, largest_suit_cards, all_discarded_cards, n_cards,pass_count_tensor], dim=1), target.unsqueeze(dim=0)
class AdvancedObserver(Observer):
    def get_observation(self, logs, player_hand: Hand) -> torch.Tensor:
        pass_count = 0
        for i in range(3):
            if logs[-(i+1)]['pass']:
                pass_count += 1
            else:
                break

        strongest_ranking = logs[-1]['strongest_ranking'] if pass_count < 3 else HandRanking.NONE
        #all discard target card 
        target = torch.concat([ranking.card_tensor.view(1, -1) 
                                for ranking in player_hand.get_all_hand_rankings_stronger_than(strongest_ranking)], 
                                dim=0) #(n_action, 1, 4, 13)

        player_card_tensor = player_hand.card_tensor.view(1, -1)
        n_action = target.size(0)


        #largest suit cards
        largest_suit_cards = strongest_ranking.card_tensor.view(1, -1) #(n_action, 1, 4, 13)
        #the number of remain hand
        n_cards = torch.zeros((1, 4, 13))
        for i in range(4):
            n_cards[:, -i, int(logs[-i]['card_tensor'].sum().item() - 1)] = 1.0

        n_cards = n_cards.view(1, -1)

        #all discard cards
        all_discarded_cards = torch.stack([log['discard'] for log in logs], dim=0).sum(dim=0).view(1, -1)

        #pass count
        pass_count_tensor = torch.zeros((1, 4))
        pass_count_tensor[:, pass_count] = 1

        remain_ranking = {}
        for ranking in Ranking:
            if ranking in [Ranking.NONE, Ranking.PASS]:
                continue
            remain_ranking[ranking] = []

        remain_card_tensors = player_hand.card_tensor  - target.view(n_action, 4, 13)
        for i in range(remain_card_tensors.size(0)):
            remain_hand = Hand(remain_card_tensors[[i]])
            for ranking in remain_ranking:
                remain_ranking[ranking].append(
                    torch.concat([torch.zeros((1, 4, 13))] + [r.card_tensor.bool() for r in remain_hand.get_hand_ranking(ranking)]).sum(dim=0, keepdim=True).float())

        n_remain_cards = nn.functional.one_hot(remain_card_tensors.sum(dim=[1, 2]).long(), 14)
        action_state = torch.concat([torch.concat(remain_ranking[Ranking.PAIR], dim=0).view(n_action, -1) for ranking in Ranking if not (ranking in [Ranking.NONE, Ranking.PASS])] + [n_remain_cards, remain_card_tensors.view(n_action, -1)], dim=1)
        observation = torch.concat([player_card_tensor, largest_suit_cards, all_discarded_cards, n_cards,pass_count_tensor], dim=1), action_state.unsqueeze(dim=0)
        return observation
    
class OrcaleObserver(Observer):
    def __init__(self, gamma = 1.0, decay = 0.9999):

        self.gamma = gamma
        self.decay = decay

    
    def get_observation(self, logs, card_tensor, largest_suit) -> torch.Tensor:
        card_tensor = Hand(card_tensor)
        #all discard target card
        target = torch.stack([suit.cards for suit in card_tensor.get_all_hand_rankings_stronger_than(largest_suit)], dim=0)
        #hand associated with discards
        remain_hand = card_tensor._card_tensor.expand_as(target) - target
        #largest suit cards
        largest_suit_cards = largest_suit.cards.expand_as(target)
        #the number of remain hand
        n_cards = torch.zeros((1, 4, 13))
        for i in range(3):
            n_cards[:, -(i+1), int(logs[-(i+1)]['card_tensor'].sum().item() - 1)] = 1.0
        n_cards = n_cards.expand_as(target)
        n_cards = torch.index_put(n_cards, 
            (torch.arange(n_cards.size(0)).view(-1, 1), 
            torch.zeros((n_cards.size(0), 1), dtype = torch.long),
            torch.zeros((n_cards.size(0), 1), dtype = torch.long),
            remain_hand.sum(dim=[1,2,3]).long().view(-1, 1)-1),
            values = torch.Tensor([1.0]))

        #all discard cards
        all_discarded_cards = torch.stack([log['discard'] for log in logs], dim=0).sum(dim=0).expand_as(target)

        observation = torch.concat([remain_hand, target, largest_suit_cards, n_cards, all_discarded_cards], dim=1).view(n_cards.size(0), -1, 13)

        #pass count
        pass_count = torch.zeros((n_cards.size(0), 1, 13))
        
        count = 0
        for i in range(3):
            if logs[-(i+1)]['pass']:
                count += 1
            else:
                break
        pass_count[:, 0, count] = 1

        oracle = torch.concat([logs[-(i+1)]['card_tensor'].expand_as(target) for i in range(3)]).view(n_cards.size(0), -1, 13)
        oracle = torch.where((oracle > 0) & (torch.rand_like(oracle) < self.gamma), 1.0, 0.0)

        observation = torch.concat([observation, pass_count, oracle], dim=1)
        self.gamma = self.gamma * self.decay
        
        return observation