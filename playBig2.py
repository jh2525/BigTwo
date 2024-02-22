import torch
import pygame
from Big2.environment import Big2Env
from Big2.setting import *
from Big2.observer import DefaultObserver
from Big2.hand import Hand
from policy import *
from model import *

env = Big2Env(DefaultObserver())
model = AC()
model.load_state_dict(torch.load('params.pth'))
agents = [ActorCriticPolicy(model)] *4

X = 1024
Y = 768
d = 50
card_width = 50
card_height = 50

suits = ['D', 'C', 'H', 'S']
suit_color = [(80, 183, 223), (170, 170, 60), (30, 230, 30), (230, 30, 30)]
selected_color =  (200, 200, 200)
no_selected_color = (230, 230, 230)

                
class Card(pygame.sprite.Sprite):
    def __init__(self, x, y, width=100, height=120, color=(0, 0, 0), text= 'S2'):
        super().__init__()

        

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.reset()

    def reset(self):
        self.original_image = pygame.Surface((self.width, self.height))
        self.original_image.fill(no_selected_color)  
        self.original_image.set_colorkey((255, 255, 255))  

        pygame.draw.rect(self.original_image, self.color, (0, 0, self.width, self.height), 2)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))

        self.original_image.blit(text_surface, text_rect)

        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
    def rotate(self, angle):
        # Rotate the image and update the rect
        self.image = pygame.transform.rotate(self.image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

class SimpleButton(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color, text, click_event = lambda : ()):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.event = click_event

        self.original_image = pygame.Surface((self.width, self.height))
        self.original_image.fill(no_selected_color)  
        self.original_image.set_colorkey((255, 255, 255))
        pygame.draw.rect(self.original_image, self.color, (0, 0, self.width, self.height), 2)
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.original_image.blit(text_surface, text_rect)
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def is_clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()

        if mouse_click[0] == 1 and not self.clicked_last_frame:
            self.clicked_last_frame = True
            return self.rect.collidepoint(mouse_pos)
        elif mouse_click[0] == 0:
            self.clicked_last_frame = False

        return False
    
    def update(self):
        if self.is_clicked():
            self.event()


class PlayerCard(Card):
    def __init__(self, x, y, width=100, height=120, color=(0, 0, 0), text= 'S2', suit = 0, number = 0):
        self.selected = False
        super().__init__(x, y, width, height, color, text)

        self.suit = suit
        self.number = number
        self.clicked_last_frame = False
        self.reset()
    def is_clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()

        if mouse_click[0] == 1 and not self.clicked_last_frame:
            self.clicked_last_frame = True
            return self.rect.collidepoint(mouse_pos)
        elif mouse_click[0] == 0:
            self.clicked_last_frame = False

        return False
    
    def reset(self):
        self.original_image = pygame.Surface((self.width, self.height))
        self.original_image.fill(selected_color if self.selected else no_selected_color)  
        self.original_image.set_colorkey((255, 255, 255))  

        pygame.draw.rect(self.original_image, self.color, (0, 0, self.width, self.height), 2)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))

        self.original_image.blit(text_surface, text_rect)

        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(topleft=(self.x, self.y))



    def update(self):
        if self.is_clicked():
            self.selected = ~self.selected
            self.reset()

    





class PlayerCards():
    def __init__(self, card_tensor: torch.Tensor):
        self.cards = []
        count = 0
        for number in range(13):
            for suit in range(4):
                if card_tensor[0, suit, number].item() > 0:
                    card = PlayerCard((-6.5+count) * card_width + X // 2, Y // 2 + 6.5 * card_height - d , 
                                card_width, card_height, color = suit_color[suit], 
                                text = str(f'{suits[suit]} {number + 2}'), 
                                suit = suit, 
                                number = number)
                    self.cards.append(card)
                    count += 1

    def get_selected(self):
        return [(card.suit, card.number) for card in self.cards if card.selected]

class DiscardedCards():
    def __init__(self, card_tensor: torch.Tensor, player = 3):
        self.cards = []
        count = 0

        x, y = [X // 2 - card_width * 2.5, Y // 2 - card_height * 2.5]
        h = card_height
        if player == 0:
            cordi = (x, y + 5*h)
            dx = h
            dy = 0
        elif player ==1:
            cordi = (x-h, y)
            dx = 0
            dy = h
        elif player == 2:
            cordi = (x + 4*h, y-h)
            dx = -h
            dy = 0
        else:
            cordi = (x+5*h, y + 4*h)
            dx = 0
            dy = -h
       

        x, y = cordi

        for number in range(13):
            for suit in range(4):
                if card_tensor[0, suit, number].item() > 0:

                    card = Card(x + dx*count, y + dy*count, card_width, card_height, color = suit_color[suit], text = str(f'{suits[suit]} {number + 2}'))
                    card.rotate(-90 * player)
                    self.cards.append(card)
                    count += 1

class OtherPlayerCards():
    def __init__(self, card_tensor: torch.Tensor, hidden = True, player = 1):
        self.cards = []
        count = 0
        d = 50
        h = card_height

        if player == 1:
            x_ = d
            y_ = Y // 2 - 6.5 * h
            dx = 0
            dy =card_height
        elif player == 2:
            x_ = X // 2 + 6.5 * h - d
            y_ = d
            dx = -h
            dy = 0
        elif player == 3:
            x_ = X - h - d
            y_ = Y // 2 + 6.5 *h - d
            dx = 0
            dy = -h

        for number in range(13):
            for suit in range(4):
                if card_tensor[0, suit, number] > 0:
                    card = Card(x_ + dx * count, y_ + dy * count , 
                                card_width, card_height, 
                                color = suit_color[suit] if not hidden else (0, 0, 0), 
                                text = str(f'{suits[suit]} {number + 2}') if not hidden else "", )
                    self.cards.append(card)
                    count += 1



class Big2RenderManager():
    def __init__(self, env, agents):
        self.sprites = []
        self.env: Big2Env = env
        self.agents = agents
        self.done = False
        self.observation, info = env.reset()
        self.chips = [50, 50, 50, 50]
    

    def clean(self):
        for sprite in self.sprites:
            sprite.kill()
        
        self.sprites.clear()

    def render(self, all_sprites):
        latest_discard = dict(zip(range(4), [torch.zeros(1, 4, 13)] * 4))
        player_hands = [self.env.board.get_card_tensor(i) for i in range(4)]
        pass_count = 0

        for i in reversed(range(len(self.env.logs))):
            if pass_count >= 3:
                
                #latest_discard[self.env.logs[i]['player']] = env.logs[i]['discard']
                break
            if self.env.logs[i]['pass']:
                pass_count += 1
            elif latest_discard[self.env.logs[i]['player']].sum() < 1e-10:
                latest_discard[self.env.logs[i]['player']] = env.logs[i]['discard']

        sprites = []
        self.player_cards = PlayerCards(player_hands[0])
        sprites += self.player_cards.cards

        for i in range(4):
            sprites += DiscardedCards(latest_discard[i], i).cards
            if not i == 0:
                sprites += OtherPlayerCards(player_hands[i], False, player = i).cards
        
        all_sprites.add(sprites)
        self.sprites = sprites

    def refresh(self,  all_sprites):
        self.clean()
        self.render(all_sprites)


    def skip(self, all_sprites):
        env = self.env

        if env.turn == 0:
            available_discard = self.env.board.get_all_hand_rankings_stronger_than(0, self.env.strongest_hand_ranking)
            if hasattr(self.agents[0], 'model'):
                with torch.no_grad():
                    critic, prob_value = self.agents[0].model(self.observation)
                    critic, prob_value = critic.detach().flatten(), prob_value.detach().flatten()

                
                prob = torch.softmax(prob_value, dim=0)

                top5_indicies = torch.topk(prob, min(5, prob.size(0))).indices
                for i in top5_indicies:
                    i = i.item()
                    print(f'{critic[i].item()}, {prob[i].item() * 100}% : {self.env.hand_to_str(available_discard[i].card_tensor)}')

                

        log_prob, value, action = self.agents[self.env.turn].choose_action(self.observation)
        next_observation, reward, done, _, info = env.step(action)
        self.observation = next_observation
        self.done = done
        if done:
            remains = [env.board.hands[i].card_tensor.sum().item() for i in range(players)]
            n_two = [env.board.hands[i].card_tensor[:, :, num_to_int['2']].sum().item() for i in range(players)]
            for toss_player in range(players):
                for accept_player in range(players):
                    amount = max(remains[toss_player] - remains[accept_player], 0) * (2 ** n_two[toss_player])
                    self.chips[toss_player] -= amount 
                    self.chips[accept_player] += amount 
            self.observation, info = env.reset()

        self.refresh(all_sprites)

    def discard(self, all_sprites):
        env = self.env
        chips = self.chips
        if self.env.turn != 0:
            return False
        
        card_tensor = torch.zeros(1, 4, 13)
        for (s, n) in self.player_cards.get_selected():
            card_tensor[0, s, n] = 1.0

        hand = Hand(card_tensor)
        available_discard = self.env.board.get_all_hand_rankings_stronger_than(0, self.env.strongest_hand_ranking)

        if hasattr(self.agents[0], 'model'):
            with torch.no_grad():
                critic, prob_value = self.agents[0].model(self.observation)
                critic, prob_value = critic.detach().flatten(), prob_value.detach().flatten()
            
            prob = torch.softmax(prob_value, dim=0)
            top5_indicies = torch.topk(prob, min(5, prob.size(0))).indices
            for i in top5_indicies:
                i = i.item()
                print(f'{critic[i].item()}, {prob[i].item() * 100}% : {self.env.hand_to_str(available_discard[i].card_tensor)}')
        action = -1

        for i in range(len(available_discard)):
            if (hand.card_tensor != available_discard[i].card_tensor).sum() < 1e-10:
                action = i

        if action == -1:
            return False

        next_observation, reward, done, _, info = self.env.step(action)
        self.observation = next_observation

        if done:
            remains = [env.board.hands[i].card_tensor.sum().item() for i in range(players)]
            n_two = [env.board.hands[i].card_tensor[:, :, num_to_int['2']].sum().item() for i in range(players)]
            for toss_player in range(players):
                for accept_player in range(players):
                    amount = max(remains[toss_player] - remains[accept_player], 0) * (2 ** n_two[toss_player])
                    self.chips[toss_player] -= amount 
                    self.chips[accept_player] += amount 
            self.observation, info = env.reset()
            
            

        
        self.refresh(all_sprites)

        
        return True
    

big2manger = Big2RenderManager(env, agents)

pygame.init()
screen = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()

big2manger.render(all_sprites)

all_sprites.add(SimpleButton(X-4*d-card_height, Y -4*d-card_height , 100, 50, (7, 26, 76), 'Pass', lambda : big2manger.skip(all_sprites)))
all_sprites.add(SimpleButton(X-4*d-card_height, Y -4*d-card_height + 50, 100, 50, (7, 26, 76), 'Discard', lambda : big2manger.discard(all_sprites)))



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill((230, 230, 230)) 
    pygame.draw.rect(screen, (0, 0, 0), (X / 2 - card_width * 2.5, Y / 2 - card_height * 2.5, card_width * 5, card_width * 5))
    pygame.draw.circle(screen, (255, 255, 0), (X/2 + card_width * 2 * np.cos(env.turn * np.pi / 2 + np.pi / 2 ), Y/2 + card_width * 2 * np.sin(env.turn * np.pi / 2 + np.pi / 2 )), 10)
    
    all_sprites.draw(screen)
    all_sprites.update()
    font = pygame.font.Font(None, 36)
    text_surface = font.render(str(big2manger.chips), True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=(X / 2, 18))
    screen.blit(text_surface, text_rect)

    if torch.cat([env.board.hands[i].card_tensor for i in range(4)]).min().item() < 0:
        break

    pygame.display.flip()


pygame.quit()