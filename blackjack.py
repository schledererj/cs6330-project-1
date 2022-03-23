from enum import Enum, auto, unique
from itertools import product
from random import choice
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
        _sum = sum([x.value for x in self._hand])
        while _sum > 21 and Rank.ACE in self._hand:
            _sum -= 10
            self._hand.remove(Rank.ACE)

        return _sum


# All gamblers are dumb
class DumbGambler(Player):
    def __init__(self, deck: Deck):
        super().__init__(deck)

    def should_hit(self):
        return self.score_hand() <= 16


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
