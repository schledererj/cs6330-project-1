from enum import Enum, auto, unique
from itertools import product
from random import choice, randrange
from copy import copy


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    FACE = 10  # pretend 10s are faces too
    ACE = 11


class RewardsTable:
    def __init__(self):
        self.table = list()
        for i in range(1, 22):
            # add an item for the action "stand"
            self.table.append({'current_state': i, 'next_state': i,
                              'action': 'stand', 'reward': i if i < 21 else 1000})
            _item = {"current_state": i}
            for j in range(1, 11):
                self.table.append({'current_state': i, 'next_state': i+j, 'action': 'hit',
                                  'reward': i+j if i+j < 21 else (1000 if i+j == 21 else -1000)})


class QTable:
    def __init__(self):
        self.table = list()
        for i in range(1, 22):
            for a in ['hit', 'stand']:
                self.table.append({'state': i, 'action': a, 'q': 0})


# assuming a continuous shuffler, i.e. chance to draw same card every draw
class Deck:
    def __init__(self):
        self._cards = (list(Rank) + ([Rank.FACE] * 3)) * 4
        self._on_deck = choice(self._cards)

    def deal(self):
        _card = copy(self._on_deck)
        self._on_deck = choice(self._cards)
        return _card

    def peek(self):
        return self._on_deck


class Player:
    def __init__(self, deck: Deck):
        self._deck = deck
        self._hand = [self._deck.deal(), self._deck.deal()]

    def hit(self):
        self._hand.append(self._deck.deal())

    def score_hand(self):
        _hand = copy(self._hand)
        _sum = sum(x.value for x in _hand)
        while _sum > 21 and Rank.ACE in _hand:
            _sum -= 10
            _hand.remove(Rank.ACE)

        return _sum


# All gamblers are dumb
class DumbGambler(Player):
    def __init__(self, deck: Deck):
        super().__init__(deck)

    def should_hit(self):
        return self.score_hand() <= 16


class QLearningGambler(Player):
    def __init__(self, deck: Deck, q_table: QTable, epsilon: float):
        super().__init__(deck)
        self.q_table = q_table
        self.epsilon = epsilon

    def should_hit(self):
        state = self.score_hand() <= 16
        hit_q = next(filter(lambda x: x['state']
                            == state and x['action'] == 'hit', self.q_table))['q']
        stand_q = next(
            filter(lambda x: x['state'] == state and x['action'] == 'stand', self.q_table))['q']

        _should_hit = hit_q > stand_q

        r = randrange(101) * 0.01
        if r < self.epsilon:
            _should_hit = choice([True, False])

        return _should_hit


class Dealer(Player):
    def __init__(self, deck: Deck):
        super().__init__(deck)

    def should_hit(self):
        return self.score_hand() <= 16


class Game:
    def __init__(self):
        self.deck = Deck()

    def play_hand(self):
        self.player = DumbGambler(self.deck)
        self.dealer = Dealer(self.deck)
        while self.player.should_hit() and self.player.score_hand() <= 21:
            self.player.hit()
        while self.dealer.should_hit() and self.dealer.score_hand() <= 21:
            self.dealer.hit()

        player_score = self.player.score_hand()
        dealer_score = self.dealer.score_hand()
        if player_score > 21:
            winner = 'dealer'
        elif dealer_score > 21:
            winner = 'player'
        elif player_score > dealer_score:
            winner = 'player'
        else:
            winner = 'dealer'
        return {
            'winner': winner,
            'player_score': player_score,
            'dealer_score': dealer_score,
            'player_cards': self.player._hand,
            'dealer_cards': self.dealer._hand
        }


class QLearningTrainer:
    def __init__(self):
        self.rewards_table = RewardsTable().table
        self.q_table = QTable().table

    def find_association_action(self, state):
        options = list(filter(lambda x: x['state'] == state, self.q_table))
        q_values = [x['q'] for x in options]
        if all(x == q_values[0] for x in q_values):
            return choice(['hit', 'stand'])
        else:
            options.sort(key=lambda x: x['q'], reverse=True)
            return options[0]['action']

    def optimize_q_table(self):
        deck = Deck()
        for i in range(1000):
            self.player = QLearningGambler(deck, self.q_table, 0.1)
            self.dealer = Dealer(self.deck)
            while self.player.should_hit() and self.player.score_hand() <= 21:
                current_state = self.player.score_hand()
                self.player.hit()
                # optimize q-table for hit action
                result_state = self.player.score_hand()
                reward = next(filter(lambda x: x['current_state'] == current_state and x['next_state']
                                     == result_state and x['action'] == 'hit', self.rewards_table))['reward']
                # assoc_action =
            # optimize q-table for stand action
            while self.dealer.should_hit() and self.dealer.score_hand() <= 21:
                self.dealer.hit()
