from time import sleep
from rl_base import Agent, State, Action
import pygame

from pygame.locals import (
    K_ESCAPE,
    K_q,
    K_w,
    K_s,
    K_d,
    K_a,
    KEYDOWN,
    QUIT,
)


class QuitException(Exception):
    pass


class ManualPygameAgent(Agent):

    def __init__(self):
        super().__init__(name="Manual Pygame Agent")

    def choose_action(self, state: State):
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) | (event.key == K_q):
                        raise QuitException()
                    elif event.key == K_w:
                        action = 0
                    elif event.key == K_s:
                        action = 1
                    elif event.key == K_a:
                        action = 3
                    elif event.key == K_d:
                        action = 2
                elif event.type == QUIT:
                    raise QuitException()
            sleep(0.05)
        return Action(action)

    def update_action_policy(self) -> None:
        pass

    def get_instruction_string(self):
        return ["w - move up", "a - move left", "d - move right", "s - move down"]

    def save(self, save_path):
        pass

    def learn(self, observation, action, reward, new_observation, done):
        pass
