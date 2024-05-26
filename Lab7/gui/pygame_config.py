import pygame

from pygame.locals import (
    RLEACCEL,
    KEYDOWN,
    K_ESCAPE,
    K_q,
    K_p,
    QUIT,
)

RED = (153, 0, 0)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
BROWN = (102, 51, 0)
BLUE_LIGHT = (128, 128, 255)
BLUE_LIGHT_2 = (153, 204, 255)
BLUE = (0, 128, 255)
GREEN = (0, 153, 0)
GREEN_DARK = (0, 102, 0)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600


BACKGROUND = BLUE_LIGHT_2


def load_assets(x_size, y_size):
    assets = {}

    pit_img = pygame.image.load("gui/assets/pit_img_2.png").convert()
    pit_img = pygame.transform.scale(pit_img, (x_size, y_size))
    assets['pit'] = pit_img

    gold_img = pygame.image.load("gui/assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (x_size, y_size))
    assets['gold'] = gold_img

    coin_img = pygame.image.load("gui/assets/free_coin.png").convert_alpha()
    coin_img = pygame.transform.scale(coin_img, (x_size, y_size))
    assets['coin'] = coin_img

    agent_img = pygame.image.load("gui/assets/agent_img.png").convert()
    agent_img = pygame.transform.scale(agent_img, (x_size, y_size))
    agent_img.set_colorkey(BLACK, RLEACCEL)
    assets['agent'] = agent_img

    agent_img_left = pygame.image.load("gui/assets/arrow_img2.png").convert_alpha()
    agent_img_left = pygame.transform.scale(agent_img_left, (x_size // 2, y_size // 2))
    assets['arrow_left'] = agent_img_left

    agent_img_down = pygame.image.load("gui/assets/arrow_img2.png").convert_alpha()
    agent_img_down = pygame.transform.scale(agent_img_down, (x_size // 2, y_size // 2))
    agent_img_down = pygame.transform.rotate(agent_img_down, 90)
    assets['arrow_down'] = agent_img_down

    agent_img_right = pygame.image.load("gui/assets/arrow_img2.png").convert_alpha()
    agent_img_right = pygame.transform.scale(agent_img_right, (x_size // 2, y_size // 2))
    agent_img_right = pygame.transform.rotate(agent_img_right, 180)
    assets['arrow_right'] = agent_img_right

    agent_img_up = pygame.image.load("gui/assets/arrow_img2.png").convert_alpha()
    agent_img_up = pygame.transform.scale(agent_img_up, (x_size // 2, y_size // 2))
    agent_img_up = pygame.transform.rotate(agent_img_up, 270)
    assets['arrow_up'] = agent_img_up

    cave_entry_img = pygame.image.load("gui/assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (x_size, y_size))
    assets['cave_entry'] = cave_entry_img

    assets['font'] = pygame.font.SysFont('Arial', 18)
    assets['q_values_font'] = pygame.font.SysFont('Arial', 15)

    return assets
