import gym
import keyboard

# noinspection PyUnresolvedReferences
import gym_marioai
from gym_marioai import levels

all_actions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
"""
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
JUMP = 4
SPEED_JUMP = 5
SPEED_RIGHT = 6
SPEED_LEFT = 7
JUMP_RIGHT = 8
JUMP_LEFT = 9
SPEED_JUMP_RIGHT = 10
SPEED_JUMP_LEFT = 11
NOTHING = 12
"""

env = gym.make('Marioai-v0', render=True,
               level_path=levels.one_cliff_level,
               compact_observation=False,
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)


def get_action():
    # return env.SPEED_JUMP_RIGHT
    # if keyboard.is_pressed('shift+space+right'):
    #     return env.SPEED_JUMP_RIGHT
    # if keyboard.is_pressed('shift+space+left'):
    #     return env.SPEED_JUMP_LEFT
    # Note: spaces is used to pause, needs to be changed
    # if keyboard.is_pressed('space+right'):
    #     return env.JUMP_RIGHT
    # if keyboard.is_pressed('space+left'):
    #     return env.JUMP_LEFT

    # NOTE: these only work with the left shift button
    if keyboard.is_pressed('shift+right'):
        return env.SPEED_RIGHT
    if keyboard.is_pressed('shift+left'):
        return env.SPEED_LEFT

    if keyboard.is_pressed('right'):
        return env.RIGHT
    if keyboard.is_pressed('left'):
        return env.LEFT
    if keyboard.is_pressed('down'):
        return env.DOWN
    if keyboard.is_pressed('up'):
        return env.JUMP
    return env.NOTHING


while True:
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        a = get_action()
        print('action', a)
        s, r, done, info = env.step(a)
        # print(len(s), ':', s)

        total_reward += r

    print(f'finished episode, total_reward: {total_reward}')

print('finished demo')
