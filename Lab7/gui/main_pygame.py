from rl_base import Agent, Env
from gui.manual_pygame_agent import QuitException, ManualPygameAgent
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import os

from gui.pygame_config import *


def key_logic(auto_mode, done):
    key_pressed = False
    running_episode = True
    last_episode = False
    for event in pygame.event.get():
        if event.type == QUIT:
            key_pressed = True
            running_episode = False
            last_episode = True
        elif event.type == KEYDOWN:
            key_pressed = True
            if event.key == K_ESCAPE or event.key == K_q:
                running_episode = False
                last_episode = True
            if event.key == K_p:
                auto_mode = not auto_mode
            if done:
                running_episode = False
    return key_pressed, running_episode, last_episode, auto_mode


def was_exploratory(q_in_state, action):
    return action != np.argmax(q_in_state)


def episode(screen, env: Env, agent: Agent, max_ep_len, i_episode, auto=False, render=True, test_mode=False):
    state = env.reset_env()

    manual_control = isinstance(agent, ManualPygameAgent)

    instruction_string = [f"Episode {i_episode}", "Goal: step onto gold", "",
                          "Instruction:", "q | ESC - terminate program"]
    if not manual_control:
        instruction_string += ["Press p to on/off auto mode", "or any other key to one step"]
    instruction_string += ["", "Agent control:"] + agent.get_instruction_string()

    if render:
        screen.fill(WHITE)
        msg = instruction_string
        if hasattr(agent, 'q_table'):
            env.render(screen, msg, agent.q_table)
        else:
            env.render(screen, msg)

    n_steps = 0
    running_episode = True
    total_reward = 0
    exploratory_moves = []
    done = False
    game_won = False
    last_episode = False
    # Main loop
    while running_episode:

        if render:
            key_pressed = False
            key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
            if auto or manual_control:
                sleep(0.05)
            else:
                while not key_pressed:
                    key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
                    sleep(0.05)

        if not done:

            try:
                action = agent.choose_action(state)
                agent.update_action_policy()

            except QuitException:
                return total_reward, n_steps, False, auto

            if action is not None:

                new_state, reward, done, info, game_won = env.step(action)

                if not test_mode:
                    agent.learn(state, action, reward, new_state, done)

                if hasattr(agent, 'q_table'):
                    exploratory_moves.append(was_exploratory(agent.q_table[state], action))

                state = new_state
                total_reward += reward

                n_steps += 1
                if render:
                    msg = instruction_string + ["", f"Agent state:"]
                    msg += ['    '+s for s in str(new_state).split(';')]
                    msg += ["", f"Reward this step: {reward}", f"Total reward: {total_reward}",
                            f"Step: {n_steps}", f"Done: {done}"]
                    msg += ["", "Info:"]
                    msg += info
                if n_steps >= max_ep_len:
                    done = True
        else:  # done
            if auto:
                break
            if render and 'end_msg' not in locals():
                end_msg = msg + ["", "Episode ended", "Press esc/q to exit or", "any other kay to start a new episode."]
                msg = end_msg

        if render:
            if hasattr(agent, 'q_table'):
                env.render(screen, msg, agent.q_table)
            else:
                env.render(screen, msg)
        else:
            if done:
                running_episode = False
    return total_reward, n_steps, not last_episode, auto, game_won, np.mean(exploratory_moves)


def main_pygame(env: Env, agent: Agent, max_ep_len=50, save_path=None, render=False,
                num_episodes=1000, info_after_episodes=50, test_mode=False):

    if not isinstance(agent, Agent):
        raise ValueError('Unsupported agent type.')

    if render:
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption('AI lab - Reinforcement Learning')
        pygame.font.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        auto_end = False
    else:
        screen = None
        auto_end = True

    i_episode = 1
    running = True
    total_rewards = []
    average_rewards = []
    game_wins = []
    average_game_wins = []
    exploration = []
    best_average_rew = -np.inf
    best_single = -np.inf
    n_steps = []
    average_n_steps = []
    while running:
        tr, ns, running, auto_end, game_won, exploration_fraction = episode(
            screen, env, agent, max_ep_len, i_episode, auto_end, render, test_mode)
        if tr > best_single:
            best_single = tr
            print(f"In {i_episode} episode, new best total_reward: {tr:05f}, in {ns} steps!")
        total_rewards.append(tr)
        avr_rew = np.mean(total_rewards[-10:])
        average_rewards.append(avr_rew)
        game_wins.append(int(game_won))
        avr_game_wins = np.mean(game_wins[-10:])
        average_game_wins.append(avr_game_wins)
        exploration.append(exploration_fraction)
        n_steps.append(ns)
        avr_steps = np.mean(n_steps[-10:])
        average_n_steps.append(avr_steps)
        if avr_rew > best_average_rew:
            best_average_rew = avr_rew
            if save_path is not None and not test_mode:
                agent.save(save_path + '_best')
            print(f"After {i_episode} episodes, new best last 10 ep. avg rew: {avr_rew:05f}, avg steps/ep: {avr_steps:.2f}")
        i_episode += 1
        if i_episode % info_after_episodes == 0:
            tmp_msg =f"After {i_episode} episodes. Last 10 avg total_rewards: {avr_rew:05f}, avg steps/ep: {avr_steps:.2f}"
            if hasattr(agent, "epsilon"):
                tmp_msg += f" eps={agent.epsilon:0.5f}"
            print(tmp_msg)
        if i_episode == num_episodes+1:
            break

    if save_path is not None and not test_mode:
        agent.save(save_path)

    if not test_mode:
        if len(average_rewards) > 10:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(average_rewards)
            axs[0].set_ylabel('Total reward')
            axs[1].plot(exploration)
            axs[1].set_ylabel('Exploration fraction')
            axs[1].set_xlabel('Episode')
            if save_path is not None and not test_mode:
                plt.savefig(os.path.dirname(save_path) + '/plots.png')
            plt.show()

    print("TEST results:")
    print(f"{np.mean(game_wins)*100:0.2f}% episodes out of {len(game_wins)} has been won.")
    print(f"Average return = {np.mean(total_rewards):0.2f}.")
