from rl_base import Env
import numpy as np
from gui.pygame_config import *


# a constant grid world (the locations of pits/wumpus objects are always the same)
# agent can take actions (4): go up, down, right, left
# state: position of the agent in the grid (which cell it is currently in)
# stochastic environment
class FrozenLake(Env):
    def __init__(self, name='FrozenLake'):
        super().__init__(name)

        self.pit_field = "P"
        self.gold_field = "G"
        self.coin_field = "c"
        self.regular_field = "F"
        self.agent_field = "A"
        self.bigger_grid = True

        self.grid_world = self.get_new_env()
        self.n_states = len(self.grid_world) * len(self.grid_world[0])

        # go up, down, right, left
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.action_names = {0: "up", 1: "down", 2: "right", 3: "left"}
        self.intended_action_prob = 0.75
        self.side_from_intended_prob = 0.1
        self.opposite_from_intended_prob = 0.05
        assert self.intended_action_prob + 2*self.side_from_intended_prob + self.opposite_from_intended_prob == 1.0, \
            "action probabilities must sum to 1.0"
        self.action_prob = {
            0: np.array([self.intended_action_prob, self.opposite_from_intended_prob, self.side_from_intended_prob, self.side_from_intended_prob]),
            1: np.array([self.opposite_from_intended_prob, self.intended_action_prob, self.side_from_intended_prob, self.side_from_intended_prob]),
            2: np.array([self.side_from_intended_prob, self.side_from_intended_prob, self.intended_action_prob, self.opposite_from_intended_prob]),
            3: np.array([self.side_from_intended_prob, self.side_from_intended_prob, self.opposite_from_intended_prob, self.intended_action_prob])
        }

        self.cave_entry_x = 0
        self.cave_entry_y = len(self.grid_world) - 1            # the entry point is the bottom left cell
        self.agentPosYX = [self.cave_entry_y, self.cave_entry_x]

        self.living_reward = -5
        self.gold_reward = 1000
        self.coin_reward = 250
        self.death_by_pit_reward = -1000

        # for rendering
        self.assets = None

    def get_new_env(self):
        if self.bigger_grid:
            return [[self.gold_field, self.regular_field, self.regular_field, self.regular_field, self.coin_field],
                    [self.pit_field, self.regular_field, self.regular_field, self.regular_field, self.regular_field],
                    [self.regular_field, self.regular_field, self.regular_field, self.regular_field, self.regular_field],
                    [self.regular_field, self.regular_field, self.regular_field, self.pit_field, self.regular_field],
                    [self.regular_field, self.regular_field, self.regular_field, self.pit_field, self.coin_field]]
        else:
            return [[self.regular_field, self.regular_field, self.gold_field, self.pit_field],
                    [self.pit_field, self.regular_field, self.regular_field, self.regular_field],
                    [self.regular_field, self.regular_field, self.regular_field, self.regular_field],
                    [self.regular_field, self.regular_field, self.regular_field, self.pit_field]]

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosYX = [self.cave_entry_y, self.cave_entry_x]
        return self.get_state(self.agentPosYX[0], self.agentPosYX[1])

    def get_state(self, y, x) -> int:
        # In this simple env state can be represented by the cell number
        return len(self.grid_world)*y + x

    def step(self, action):
        if action not in self.action_space:
            raise Exception("Invalid action")

        reward = self.living_reward
        info = []
        game_won = False

        # stochastic action
        intended_action = action
        action = np.random.choice([0, 1, 2, 3], p=self.action_prob[intended_action])

        if action == intended_action:
            info.append(f"Agent moved {self.action_names[intended_action]}.")
        else:
            info.append(f"Agent wanted {self.action_names[intended_action]}, but moved {self.action_names[action]}.")

        # up
        if action == 0:
            self.agentPosYX[0] -= 1
            if self.agentPosYX[0] < 0:
                self.agentPosYX[0] = 0
        # down
        elif action == 1:
            self.agentPosYX[0] += 1
            if self.agentPosYX[0] > len(self.grid_world) - 1:
                self.agentPosYX[0] = len(self.grid_world) - 1
        # right
        elif action == 2:
            self.agentPosYX[1] += 1
            if self.agentPosYX[1] > len(self.grid_world[self.agentPosYX[0]]) - 1:
                self.agentPosYX[1] = len(self.grid_world[self.agentPosYX[0]]) - 1
        # left
        elif action == 3:
            self.agentPosYX[1] -= 1
            if self.agentPosYX[1] < 0:
                self.agentPosYX[1] = 0

        new_state = self.get_state(self.agentPosYX[0], self.agentPosYX[1])

        done = False
        if self.grid_world[self.agentPosYX[0]][self.agentPosYX[1]] == self.regular_field:
            pass
        elif self.grid_world[self.agentPosYX[0]][self.agentPosYX[1]] == self.coin_field:
            reward += self.coin_reward
            self.grid_world[self.agentPosYX[0]][self.agentPosYX[1]] = self.regular_field
            info.append("Found coin.")
        else:
            done = True
            if self.grid_world[self.agentPosYX[0]][self.agentPosYX[1]] == self.gold_field:
                reward += self.gold_reward
                info.append("Found gold, you won.")
                game_won = True
            elif self.grid_world[self.agentPosYX[0]][self.agentPosYX[1]] == self.pit_field:
                reward += self.death_by_pit_reward
                info.append("Fell into pit, you lost.")
            else:
                info.append("Strange ending?")

        return new_state, reward, done, info, game_won

    def render(self, screen, text, q_values=None):
        x_size = SCREEN_HEIGHT // len(self.grid_world[0])
        y_size = SCREEN_HEIGHT // len(self.grid_world)
        if not self.assets:
            self.assets = load_assets(x_size=x_size, y_size=y_size)

        screen.fill(WHITE)

        for y in range(len(self.grid_world)):
            for x in range(len(self.grid_world[y])):
                if self.grid_world[y][x] == self.regular_field:
                    pygame.draw.rect(screen, BACKGROUND, pygame.Rect(x * x_size, y * y_size, x_size, y_size))
                elif self.grid_world[y][x] == self.gold_field:
                    screen.blit(self.assets['gold'], (x * x_size, y * y_size, x_size, y_size))
                elif self.grid_world[y][x] == self.coin_field:
                    pygame.draw.rect(screen, BACKGROUND, pygame.Rect(x * x_size, y * y_size, x_size, y_size))
                    screen.blit(self.assets['coin'], (x * x_size, y * y_size, x_size, y_size))
                elif self.grid_world[y][x] == self.pit_field:
                    screen.blit(self.assets['pit'], (x * x_size, y * y_size, x_size, y_size))

                best = None
                if q_values is not None and (self.grid_world[y][x] == self.regular_field):

                    state = self.get_state(y, x)
                    sorted_actions = np.argsort(q_values[state])
                    if q_values[state, sorted_actions[3]] > q_values[state, sorted_actions[2]]:  # there is a clear best action
                        best = sorted_actions[3]

                    arrows = {0: self.assets['arrow_up'],
                              1: self.assets['arrow_down'],
                              2: self.assets['arrow_right'],
                              3: self.assets['arrow_left']}
                    if best is not None:
                        screen.blit(arrows[best], (x * x_size + x_size//4,
                                                   y * y_size + y_size//4,
                                                   x_size//2, y_size//2))

                # draw agent
                if (y == self.agentPosYX[0]) & (x == self.agentPosYX[1]):
                    screen.blit(self.assets['agent'],
                                (x * x_size, y * y_size, x_size, y_size))

                pygame.draw.rect(screen, BLUE, pygame.Rect(x * x_size, y * y_size,
                                                           x_size, y_size), 5)

                # draw q-values at the edges of each field
                if q_values is not None and self.grid_world[y][x] == self.regular_field:

                    def set_position(msg_rect, action):
                        if action == 0:  # 'up'
                            msg_rect.midtop = ((x+0.5)*x_size, y*y_size + 2)
                        elif action == 1:  # 'down'
                            msg_rect.midbottom = ((x+0.5)*x_size, (y+1)*y_size - 2)
                        elif action == 2:  # 'right'
                            msg_rect.midright = ((x+1)*x_size - 5, (y+0.5)*y_size)
                        elif action == 3:  # 'left'
                            msg_rect.midleft = (x*x_size + 5, (y+0.5)*y_size)
                        else:
                            raise ValueError('bad action value')
                        return msg_rect

                    for a in range(4):
                        color = GREEN_DARK if best is not None and a == best else RED
                        msg = self.assets['q_values_font'].render(f"{q_values[self.get_state(y, x), a]:04.1f}", False, color)
                        screen.blit(msg, set_position(msg.get_rect(), a))

        for t in range(len(text)):
            msg = self.assets['font'].render(text[t], False, BLACK)
            screen.blit(msg, (x_size * len(self.grid_world[0]) + 10, t * 25))

        pygame.display.flip()
