from Big2.setting import *
from Big2.board import Board
from Big2.hand import Hand
from Big2.handranking import HandRanking, Ranking
from Big2.observer import Observer


class Big2Env():
    def __init__(self, observer: Observer, render = False):
        self.board: Board = Board()
        self.logs = []
        self.observer = observer
        self.strongest_hand_ranking: HandRanking
        self.pass_count = 0

    def reset(self, seed: Optional[int] = None, player_fix_cards: Optional[np.ndarray] = None):
        if player_fix_cards is None:
            player_fix_cards = np.zeros((4, 4*13))
        self.board.initialize(seed, player_fix_cards)

        #find the player who discards cards at first
        for i in range(players):
            if self.board.hands[i].card_tensor[0][sn_index('diamond' , '3')].item() > 0:
                self.turn = i
                break
        
        self.pass_count = 0
        self.strongest_hand_ranking = HandRanking.NONE
        self.logs = []
        for i in range(4):
            self.append_log(
                player = (self.turn + i)  % players,
                card_tensor = self.board.hands[(self.turn + i) % players].card_tensor,
                discard = torch.zeros(1, 4, 13),
                act_pass = False,
                strongest_ranking = HandRanking.NONE
            )
        observation = self.observer.get_observation(self.logs, self.board.hands[self.turn])
        

        return observation, {}
    
    def step(self, action: int):
        
        available_rankings = self.board.get_all_hand_rankings_stronger_than(self.turn, self.strongest_hand_ranking)
        action_ranking = available_rankings[action]

        discard = action_ranking.card_tensor
        act_pass = (action_ranking.ranking == Ranking.PASS)

        if act_pass:
            self.pass_count += 1
        else:
            self.pass_count = 0
            self.strongest_hand_ranking = action_ranking
        
        self.append_log(
            player = self.turn, 
            card_tensor = self.board.get_card_tensor(self.turn),
            discard = discard, 
            act_pass = act_pass, 
            strongest_ranking = self.strongest_hand_ranking
        )

        terminated = False
        info = {}
        reward = discard.sum().item() / 5.0
        truncated = False
        next_observation = None

        #if passed 3 times in a row
        if self.pass_count >= 3:
            self.strongest_hand_ranking = HandRanking.NONE
            self.pass_count = 0

        #discard
        if not act_pass:
            self.board.discard_cards(self.turn, discard)
            if self.board.get_n_card(self.turn) < 1:
                terminated = True
        
        #turn next player
        self.turn = (self.turn + 1) % players
        next_observation = self.observer.get_observation(
            logs = self.logs, 
            player_hand = self.board.hands[self.turn], 
        )
        
        return next_observation, reward, terminated, truncated, info
    

    def append_log(self, player, card_tensor, discard, act_pass, strongest_ranking):
        self.logs.append({'player':player, 'card_tensor':card_tensor, 'discard':discard, 'pass':act_pass, 'strongest_ranking':strongest_ranking})

        return self
    
    def hand_to_str(self, card_tensor: torch.Tensor):
        shape = ['D', 'C', 'H', 'S']
        result = ''
        for i in range(4):
            numbers = [str(i.item()) for i in torch.where(card_tensor[0][i] > 0)[0] + 2]
            if len(numbers) > 0:
                result += shape[i]
            for n in numbers:
                result += n + ' '
        return result
        