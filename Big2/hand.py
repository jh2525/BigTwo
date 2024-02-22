
from Big2.setting import *
from typing import *
from Big2.handranking import *
from itertools import chain


def get_n_common_cards(card_tensor: torch.Tensor, n: int, transpose: bool = False) -> List[torch.Tensor]:
    if transpose:
        card_tensor = card_tensor.permute(0, 2, 1)

    n_sums = card_tensor.sum(dim=1)
    _, numbers = torch.where(n_sums >= n)

    result = []

    for i in numbers:
        indicies,  = torch.where(card_tensor[0, :, i])
        cases = torch.combinations(indicies, n)
        for j in range(len(cases)):
            n_same_cards = torch.zeros_like(card_tensor)
            n_same_cards[0, cases[j], i] = 1.0
            
            if transpose:
                n_same_cards = n_same_cards.permute(0, 2, 1)
            
            result.append(n_same_cards)
        
    return result


def get_n_m_common_cards(card_tensor: torch.Tensor, n: int, m: int) -> List[torch.Tensor]:
    result = []
    n_common_cards = get_n_common_cards(card_tensor, n)
    for n_card in n_common_cards:
        m_common_cards = get_n_common_cards(card_tensor - n_card, m)
        for m_card in m_common_cards:
            result.append(n_card + m_card)
    return result


straight_filter = torch.full((1, 1, 5,), 1.0)

def get_straight(card_tensor: torch.Tensor) -> List[torch.Tensor]:
    concat_tensor = torch.concat([card_tensor[:, :, -1:], card_tensor], dim=-1)
    column_sum = (concat_tensor.sum(dim=1) > 0).float()
    straight_num = torch.conv1d(column_sum, straight_filter)
    straight_indicies = torch.where(straight_num.squeeze() > 4)[0]

    result = []

    for i in straight_indicies:
        if i == 0: #in case of back straight
            straight_sequences = torch.Tensor([-1, 0, 1, 2, 3]).long()
        else:
            straight_sequences = torch.arange(i-1, i+4)

        straight_target = torch.cartesian_prod(
            *[torch.where(concat_tensor[0, :, i+j])[0] for j in range(5)])
        
        for j in range(len(straight_target)):
            straight_tensor = torch.zeros_like(card_tensor)
            straight_tensor[:, straight_target[j], straight_sequences] = 1.0
            result.append(straight_tensor)

    return result
        
        
class Hand():
    def __init__(self, card_tensor: Optional[torch.Tensor] = None):
        if card_tensor is None:
            card_tensor = torch.zeros(1, 4, 13)
        self._card_tensor = card_tensor
        self._hand_rankings: Dict[Ranking, HandRanking] = {}
        
        self.update()
    
    def get_n_card(self):
        return int(self._card_tensor.sum().item())

    
    def get_card_tensor(self):
        return self._card_tensor.clone()
    
    def update(self):
        card_tensor = self._card_tensor

        single = get_n_common_cards(card_tensor, 1)
        pair = get_n_common_cards(card_tensor, 2)
        triple = get_n_common_cards(card_tensor, 3)

        flush = get_n_common_cards(card_tensor, 5, True)
        four_card = get_n_m_common_cards(card_tensor, 4, 1)
        full_house = get_n_m_common_cards(card_tensor, 3, 2)
        straight = get_straight(card_tensor)
        
        if len(straight) == 0:
            straight_flush = []
        else:
            straight_flush_indices = torch.where(torch.max(torch.concat(straight).sum(dim=-1), dim=-1).values == 5)[0].tolist()
            straight_flush = [straight[i] for i in straight_flush_indices]
        
        if len(straight_flush) > 0 and len(flush) > 0:
            straight_flush_indices.reverse()
            for i in straight_flush_indices:
                del straight[i]

            concat_flush = torch.concat(flush)
            indicies = []
            for sf in straight_flush:
                indicies.append(torch.where((sf.expand_as(concat_flush) * concat_flush).view(concat_flush.size(0), -1).sum(dim=-1) == 5)[0].item())
                
            for i in sorted(indicies, reverse = True):
                del flush[i]
        
        for (ranking_tensors, ranking) in zip([None, single, pair, triple, straight, flush, full_house, four_card, straight_flush], Ranking):
            if ranking is Ranking.NONE:
                continue
            self._hand_rankings[ranking] = [HandRanking(card_tensor, ranking) for card_tensor in ranking_tensors]
            
        return self
    
    def discard_cards(self, card_tensor: torch.Tensor):
        self._card_tensor = self._card_tensor - card_tensor
        self.update()
        return self
    
    def get_hand_ranking(self, ranking: Ranking):
        return self._hand_rankings[ranking]
    def get_all_hand_rankings(self) -> List[HandRanking]:
        return  [HandRanking.PASS] + list(chain(*self._hand_rankings.values()))
    
    def get_all_hand_rankings_stronger_than(self, target_ranking: HandRanking) -> List[HandRanking]:
        results = []
        for hand_ranking in self.all_hand_rankings:
            if target_ranking < hand_ranking:
                results.append(hand_ranking)
        return results
    
    card_tensor = property(get_card_tensor)
    all_hand_rankings = property(get_all_hand_rankings)
    n_card = property(get_n_card)
    
        