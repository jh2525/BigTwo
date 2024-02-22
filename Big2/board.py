from Big2.hand import Hand
from Big2.setting import *
from Big2.handranking import HandRanking

class Board():
    def __init__(self):
        self.initialize()
    
    def initialize(self, seed: Optional[int] = None, player_fix_cards: Optional[np.ndarray] = None):
        if player_fix_cards is None:
            player_fix_cards = np.zeros((4, 4*13))
        if not seed is None:
            np.random.seed(seed)
        
        n_fix_card = player_fix_cards.sum(axis=1)
        fix_cards = player_fix_cards.sum(axis=0)

        shuffled_cards = np.arange(13*4)[~fix_cards.astype(bool)]
        np.random.shuffle(shuffled_cards)

        hands = np.zeros((4, 52)) + player_fix_cards

        for i in range(4):
            start = 0 if i == 0 else end
            end = 13 - int(n_fix_card[i]) + (end if i != 0 else 0)
            hands[i, shuffled_cards[start:end]] = 1

        self.hands : List[Hand] = [Hand(torch.from_numpy(hands[i]).float().view(1, 4, 13)) for i in range(players)]

        return self
    
    def get_card_tensor(self, player: int) -> torch.Tensor:
        return self.hands[player].card_tensor
    
    def get_n_card(self, player: int) -> int:
        return self.hands[player].n_card
    
    def get_all_rankings(self, player: int) -> List[HandRanking]:
        return self.hands[player].get_all_hand_rankings()
    
    def discard_cards(self, player: int, card_tensor: torch.Tensor):
        self.hands[player].discard_cards(card_tensor)
        return self

    def get_all_hand_rankings_stronger_than(self, player: int, hand_suit: HandRanking) -> List[HandRanking]:
        return self.hands[player].get_all_hand_rankings_stronger_than(hand_suit)
