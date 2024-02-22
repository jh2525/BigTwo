from Big2.setting import *
from enum import Enum

class Ranking(Enum):
    NONE = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_CARD = 7
    STRAIGHT_FLUSH = 8
    PASS = 9

power_matrix = torch.arange(0, max_num * n_suit).view(max_num, n_suit).T.roll(1, dims=1) 

def get_maximum_number_power(hand_tensor: torch.Tensor) -> float:
    """
    인자로 주어진 손패 행렬에 대해서, 가장 강한 끗수와 슈트를 대소가 바뀌지 않는 숫자로 반환합니다.
    """
    if 3 != hand_tensor.dim():
        raise ValueError
    _, suits, numbers = torch.where(hand_tensor)
    return power_matrix[suits, numbers].max().item()


class HandRanking:

    def __init__(self, card_tensor: torch.Tensor, ranking: Ranking):
        self._card_tensor = card_tensor #손패 행렬
        self._ranking = ranking #족보 이름
        if ranking in [Ranking.SINGLE, Ranking.PAIR, Ranking.TRIPLE, Ranking.STRAIGHT, Ranking.STRAIGHT_FLUSH, Ranking.FLUSH]:
            self.power = get_maximum_number_power(card_tensor) #싱글, 페어, 트리플, 스트레이트, 스트레이트 플러시인 경우는 단순히 가장 강한 끗수와 슈트를 비교하기만 하면된다.
        elif ranking in [Ranking.FULL_HOUSE, Ranking.FOUR_CARD]: #풀하우스나, 포카드인 경우는 가장 많은 사용한 끗수의 가장 강한 슈트와 끗수를 비교해야한다.
            target = torch.zeros_like(card_tensor)
            target_indicies = card_tensor.sum(dim=1).argmax()
            target[:, :, target_indicies] = card_tensor[:, :, target_indicies] #가장 많이 사용한 끗수의 카드들만 1로 만들어서 족보를 구성하는 가장 강한 끗수와 슈트를 구한다.
            self.power = get_maximum_number_power(target)
        else:
            self.power = 0 #None 또는 Pass인 경우는 항상 power가 0으로 정의한다.

    def __lt__(self, other):
        ranking_value = self._ranking.value
        if self._ranking == Ranking.NONE:
            return (other.ranking != Ranking.PASS)
        elif other.ranking == Ranking.PASS:
            return True
        elif ranking_value <= 3:
            return (ranking_value == other.ranking.value) and (self.power < other.power)
        else:
            if ranking_value < other.ranking.value:
                return True
            else:
                return (ranking_value == other.ranking.value) and (self.power < other.power)

    def get_card_tensor(self):
        return self._card_tensor.clone()
    
    def get_ranking(self):
        return self._ranking
    
    card_tensor = property(get_card_tensor)
    ranking = property(get_ranking)

HandRanking.PASS = HandRanking(torch.zeros((1, 4, 13)), Ranking.PASS) #Pass 족보는 항상 똑같기 때문에 정적 변수로 생성해준다
HandRanking.NONE = HandRanking(torch.zeros((1, 4, 13)), Ranking.NONE) #None 족보도 마찬가지