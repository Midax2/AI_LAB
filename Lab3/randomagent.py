import random

from exceptions import AgentException


class RandomAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return random.choice(connect4.possible_drops())
