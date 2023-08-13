#!/usr/bin/env python
# coding: utf-8

# # Training an Agent to play Super Mario with Offline Learning
# ---
#
# In this exercise you will learn how to use offline learning to train a neural network to play Super Mario.
# Its performance will then be compared with the results from the Q-Learning exercise.

# ## 0. Setup
#
# ### Requirements
#  - Java 8 (or later) runtime environment
#  - Python 3.6 (or later)
#  - Microsoft Visual C++ 14.0 (or later)
#
# You will be provided with both the .jar and the gym-marioai python package.

# ### Installation
# To setup this exercise we will use Pipenv.
# If you do not have pipenv installed please do so by running:
#
# ``pip install --user pipenv``
#
# After that please run (if not already done)
#
# ``pipenv install`` and
# ``pipenv run start``.
#
#
# in the root directory to start the Jupyter notebook containing the exercise.
#
# (pipfile set to python version 3.6 by default, can be edited if needed)

# In[15]:


import pathlib

import d3rlpy.dataset
from d3rlpy.algos import DQN
# Setup the imports. Run this cell again if you encounter any import errors.
import numpy as np
from d3rlpy.dataset import MDPDataset, ReplayBuffer

from controller import GamepadController, KeyboardController
from gym_setup import Env

from data.datasets.getDatasets import getDataset
from sklearn.model_selection import train_test_split

# from d3rlpy.metrics.scorer import evaluate_on_environment
# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('set_env', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python')


# ## 1. Generating data
# First, you will have to generate some data for the neural network to train with.
#
# ### Level
# The level we will be using for this exercise will be a very easy one to minimize the training time.
#
# However, if you would like to try different levels, we encourage you to do so by changing the ``level`` variable below to a different one from the ``levels`` folder.

# In[16]:


# Setup global variables
init_dir = pathlib.Path("gym-marioai/gym_marioai").resolve()
level = init_dir / pathlib.Path("levels", "OneCliffLevel.lvl")
dataset_path = init_dir / pathlib.Path("data", "datasets", level.stem + ".h5")
dataset_path_rand = init_dir / pathlib.Path("data", "datasets", level.stem + ".random.h5")

print(level)

# ### 1.1 Player generated data.
#
# ![Mario](https://media1.giphy.com/media/aX0RqLt2ARSW4/giphy.gif?cid=ecf05e47fnkts3fqh25tj9v8noh9vnccwo4x0ey4zpdxc7ft&rid=giphy.gif&ct=g)
#
#
# To achieve the best possible results, the training algorithm needs the best possible data. In this case that means player generated data.
#
#
# You will have the most fun playing with a USB-Controller but if you have none, you can set the following variable to ``False`` to use the keyboard:

# In[17]:


# Don't forget to run me
USE_GAMEPAD = False

# ### Controls
# |            	| Keyboard    	| Xbox       	| Playstation 	|
# |------------	|-------------	|------------	|-------------	|
# | Jump       	| S           	| A          	| X           	|
# | Sprint     	| A           	| B          	| O           	|
# | Move Right 	| Arrow Right 	| Dpad Right 	| Dpad Right  	|
# | Move Left  	| Arrow Left  	| Dpad Left  	| Dpad Left   	|
# | Duck       	| Arrow Down  	| Dpad Down  	| Dpad Down   	|

# To start the game run the next cell. If you think you have enough data just close the game window and move on to the next cell.
#
# Note: We have pregenerated some training data for your convenience which will be used in addition to your data to train the model.
# If you want to train with your own data only, go ahead and delete the data from ``exercise_offline_rl\data\datasets``.

# In[18]:

"""
# Let's play!
try:
    env_play = Env(visible=True, level=str(level)).env
    if USE_GAMEPAD:
        controller = GamepadController(env_play)
    else:
        controller = KeyboardController(env_play)
    while True:
        observation = env_play.reset()
        done = False
        action = controller.read()

        observations = [observation]
        actions = [action]
        # No reward at first time step, because no action was taken yet
        rewards = [0]
        terminals = [done]

        while not done:
            observation, reward, done, info = env_play.step(action)
            action = controller.read()

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)

        dataset = None
        if dataset_path.exists():
            # NOTE: is this the correct type of buffer?
            # see https://d3rlpy.readthedocs.io/en/latest/references/dataset.html
            with dataset_path.open("rb") as dataset_file:
                dataset: ReplayBuffer = ReplayBuffer.load(dataset_file, d3rlpy.dataset.InfiniteBuffer())
                dataset.append_episode(d3rlpy.dataset.components.Episode(np.asarray(observations), np.asarray(actions),
                               np.asarray(rewards), done))
        else:
            dataset = MDPDataset(np.asarray(observations), np.asarray(actions),
                                 np.asarray(rewards), np.asarray(terminals))
        with open(dataset_path, "w+b") as f:
            dataset.dump(f)
except ConnectionResetError:
    print("Done")

exit()
"""

# ### 1.2 Randomly generated data (optional)
# To complement the player generated data, it is possible to also generate some random data for the algorithm to train with.

# In[ ]:


# Generate random data
"""
EPISODES = 2  # <--- increase if you want more random data. More data might slow down the training process.

env_rand = Env(visible=False, level=str(level)).env

for episode in range(EPISODES):
    observation = env_rand.reset()
    done = False
    action = env_rand.action_space.sample()

    observations = [observation]
    actions = [action]
    # No reward at first time step, because no action was taken yet
    rewards = [0]
    terminals = [done]

    while not done:
        observation, reward, done, info = env_rand.step(action)
        action = env_rand.action_space.sample()

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)

    dataset = None
    if dataset_path_rand.exists():
        # NOTE: is this the correct type of buffer?
        # see https://d3rlpy.readthedocs.io/en/latest/references/dataset.html
        with dataset_path_rand.open("rb") as dataset_file:
            dataset: ReplayBuffer = ReplayBuffer.load(dataset_file, d3rlpy.dataset.InfiniteBuffer())
            dataset.append_episode(d3rlpy.dataset.components.Episode(np.asarray(observations), np.asarray(actions),
                           np.asarray(rewards), done))
    else:
        dataset = MDPDataset(np.asarray(observations), np.asarray(actions),
                             np.asarray(rewards), np.asarray(terminals))

    dataset.dump(dataset_path_rand)

print("Done!")
"""

# ## 2. Use the generated data to train a policy
# Now that you have generated some data for the neural network to train with, let's begin with the training.
# For the purpose of this exercise we will use the Offline RL Python library [d3rlpy](https://github.com/takuseno/d3rlpy).
#
#

# ### 2.1 Choosing an algorithm
# ![DQN](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/dqn.PNG)
#
# #### Why we chose DQN
# The Deep Q-Network approach is known to have been able to achieve human-level control in Atari games, and it is able to learn successful policies directly from high-dimensional sensory inputs (like pixels) using end-to-end reinforcement learning which makes it ideal for our purpose. [[1]](https://www.nature.com/articles/nature14236)
#
# It uses data collected from an environment to learn and train without interacting with it.
#
# For more information on DQN, please refer to [this paper](https://www.nature.com/articles/nature14236).
#
#

# ### 2.2 Setup the training
# First we set the training parameters.
# Most of these are fairly optimized but feel free to experiment.

# In[ ]:


# Training parameters
n_epochs = 100  # <--- change here if you want to train more / less
test_size = 0.1  # percentage of episodes not used for training

# DQN parameters
learning_rate = 0.0003  # to what extent the agent overrides old information with new information
gamma = 0.99  # discount factor, how important future rewards are
target_update_interval = 3000  # interval of steps that the agent uses to update target network
batch_size = 2  # size of training examples utilized in one iteration
use_gpu = False  # usage of gpu to train

# ### 2.3 Training time!
#
# If you want to track the training with tensorboard, run the following cell.

# In[ ]:


# Start tensorboard
# get_ipython().run_line_magic('tensorboard', '--logdir runs')

# To start the training run the next cell:

# In[ ]:


dataset = getDataset()
train_episodes, test_episodes = dataset, dataset
# train_episodes, test_episodes = train_test_split(dataset, test_size=test_size)

dqn = DQN(learning_rate=learning_rate, gamma=gamma,
          target_update_interval=target_update_interval,
          batch_size=batch_size, use_gpu=use_gpu)
dqn.build_with_dataset(dataset)

# set environment in scorer function
env_train = Env(visible=False, level=str(level)).env
evaluate_scorer = evaluate_on_environment(env_train)
# evaluate algorithm on the environment
rewards = evaluate_scorer(dqn)
name = 'marioai_%s_%s_%s_%s_%s' % (level.stem, gamma, learning_rate, target_update_interval, n_epochs)
model_file = pathlib.Path("data", "models", name + ".pt")
currentMax = -100000
dqn_max = copy.deepcopy(dqn)

fitter = dqn.fitter(train_episodes, eval_episodes=test_episodes,
                    tensorboard_dir='runs', experiment_name=name,
                    n_epochs=n_epochs, scorers={'environment': evaluate_scorer})

for epoch, metrics in fitter:
    if metrics.get("environment") > currentMax:
        currentMax = metrics.get("environment")
        dqn_max.copy_q_function_from(dqn)
    else:
        dqn.copy_q_function_from(dqn_max)
    dqn.save_model(model_file)
    if currentMax > 100:
        # For the purpose of the exercise the training will stop if the agent manages to complete the level
        print("A suitable model has been found.")
        break

# ### 2.4 Validation
# Now let's see if the training did something. If the results are not as expected try recording more data or increasing the training epochs.

# In[ ]:


env_show = Env(visible=True, level=str(level)).env
dqn = DQN()
dqn.build_with_dataset(getDataset())
dqn.load_model(model_file)

try:
    while True:
        observation = env_show.reset()
        done = False
        total_reward = 0
        while not done:
            observation, reward, done, info = env_show.step(dqn.predict([observation])[0])
            total_reward += reward
        print(f'finished episode, total_reward: {total_reward}')
except ConnectionResetError:
    print("Window closed.")

# ## 3. Offline RL vs Online RL
#
# Now we want to compare the approach from the first exercise where an online Q-Learner was used with the results we were able to achieve with the offline RL approach demonstrated above.
#
# ![Gif](https://1.bp.blogspot.com/-O0FvK3zJd9w/XpXqiJduwyI/AAAAAAAAFtM/5hxzdWOoSLw5sd5vEgMsiGVJSATKx1oEgCLcBGAsYHQ/s640/OFFLINE%2BRL%2Bfig1%2B05b.gif)
#

# ### 3.1 Reproducibility
# To compare as fairly as possible we ran both the Online Q-Learner as well as the offline Deep Q-Network until a plateau of performance has been reached.
# - The online Q-Learner was able to train for 10.000 episodes per level while being able to interact with the environment
# - The DQN was fed <1 hour (~360 episodes) of human playtime per level and was not allowed to interact with the environment while training
# - Both models were using the same reward settings
#

# ### 3.2 Easy level
#
# |  | Reward | Video |
# | -------- | -------- | -------- |
# | Q-Learner  | 232     | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/rough_terrain_q_learner_232.gif)    |
# | Deep Q-Network  | 280    | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/rough_terrain_dqn_280.gif)    |
#
# With sufficient training, neither model struggles with the easy level. However, the model fed with player generated data shows better anticipation of jumps which leads to a better overall result.

# ### 3.3 Medium level
#
# |  | Reward | Video |
# | -------- | -------- | -------- |
# | Q-Learner  | 176     | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/cliff_and_enemies_q_learner_176.gif)    |
# | Deep Q-Network  | 193    | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/cliff_and_enemies_dqn_193.gif)    |
#
# In the medium level, both models behave similarly to the easy level. While the Online Learner seems to have a better strategy to avoid enemies, the Offline Learner has the better jumping performance leading to a better overall score as it is quicker to finish the level.

# ### 3.4 Hard level
# |  | Reward | Video |
# | -------- | -------- | -------- |
# | Q-Learner  | -559     | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/climb_q_learner_-559.gif)    |
# | Deep Q-Network  | -34    | ![Gif](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/climb_dqn_-34.gif)    |
#
# Neither model manages to complete the really hard level we tested them on. It is however interesting to observe the different strategies they applied. While the Q-Learner shows a very promising leap to the middle platform, failing to reach the final platform it seems to just give up and wait for the time to run out leading to a high time punishment and therefore an extremely low score. The offline trained model while still failing to complete the level, has developed a strategy to avoid the high time punishment by committing suicide as soon as possible.

# ### 3.5 Reward Summary
#
# ![Summary](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/level-summary.png)
#
#
#

# ### 3.6 Performance over 100 levels
#
# To compare how well the different methods are able to generalize, we let both the online Q-Learner and the offline DQN train on a single level (the easy level above) and then let them play 100 randomly generated levels (including the three levels above).
#
# The model obtained by Q-Learning is only able to beat a single level (presumably the one it was trained on) which indicates that the model highly overfits the level used for training.
#
# The model produced by the DQN using user-supplied data on the other hand is able to beat 12 out of the 100 levels and manages to obtain a positive reward on 21 of them. While that number may still not be terribly impressive, it shows that the offline approach is able to generalize good enough to beat levels it has never seen before and far surpasses what the Q-Learner was able to achieve.
#
# The plot below shows the total reward achieved by the offline and online method for each level. Here we can also see that the offline method is usually able to obtain a higher reward than the online method. This means that either the offline agent is able to progress further into the level or that the online learner gets stuck somewhere while the offline learner prefers a quick death like in the hard level above.
#
# ![Comparison: Achieved Reward](https://raw.githubusercontent.com/koerners/marioai/master/exercise_offline_rl/data/jupyter/eval.png)
#

# ## 4. Conclusion
#
# While achieving slightly better results than the online RL approach in the easy and medium level, the offline DQN approach was not able to perform any better than the online implementation in the hard level we tested them both on.
# Our guess would be the lack of training data as mastering a level of this complexity would require a level of understanding of the environment that the network was not able to abstract from the data we were able to provide.
# This also demonstrates the biggest issue with the offline RL approach: it's high dependence on good and plentiful data.
# Creating such data is a time-consuming and often times an expensive endeavor.
#
# Nevertheless, even with limited data the offline approach was still able to generalize better than the online approach, as we have learned from our evaluation on random levels.
# We also believe that there is room to improve for the offline approach even without providing additional data.
# For example, we expect that the training results with our limited amount of data might be improved by adding a Convolutional Neural Network to the DQN.
# This might lead to a precise Pixel recognition of the matrix that our
# agent perceives thus leading to better reactions in certain situations with the training data of one balanced level which the player faces different situation.
#
#
# In the real world offline RL is often times the only sensible way to use an ML model as online learning would simply be too dangerous or slow.
# However, this comes with the price of the agent not being able to explore different approaches on its own meaning an often times subpar training result.
# Companies like Tesla have acknowledged this issue and have been training their self-driving cars in a life like simulated environment in which they reproduce difficult situations the car might encounter to explore without any real world damages. [[3]](https://youtu.be/11QXiJ8ORe8?t=3187)

# ![Thanks](https://media4.giphy.com/media/1mssFONYwmBlJy1DAv/giphy.gif?cid=ecf05e47fq7b3e8nbn49rxb2hj1f8qy627umny603h7tsi8f&rid=giphy.gif&ct=g)
