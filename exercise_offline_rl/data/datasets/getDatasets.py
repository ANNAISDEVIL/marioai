import os
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer


# def getDataset():
#     dataset = None
#     directory = os.path.dirname(os.path.realpath(__file__))
#     for entry in os.scandir(directory):
#         if entry.path.endswith(".h5") and entry.is_file():
#             print(entry.path)
#             if dataset is not None:
#                 # dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
#                 for episode in ReplayBuffer.load(entry.path, InfiniteBuffer()).episodes:
#                     dataset.append_episode(episode)
#             else:
#                 dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
#     return dataset

# def getDataset(training_mode=None):
#     dataset = None
#     directory = os.path.dirname(os.path.realpath(__file__))
    
#     for entry in os.scandir(directory):
#         if entry.path.endswith(".h5") and entry.is_file():
#             if training_mode == None:
#                 print(entry.path)
#             elif training_mode == "ClimbLevel" and "ClimbLevel" in entry.name:
#                 print(entry.path)
#             elif training_mode == "OneCliffLevel" and "OneCliffLevel" in entry.name:
#                 print(entry.path)
            
#             if dataset is not None:
#                 for episode in ReplayBuffer.load(entry.path, InfiniteBuffer()).episodes:
#                     dataset.append_episode(episode)
#             else:
#                 dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
#     return dataset

def getDataset(training_mode=None):
    dataset = None
    directory = os.path.dirname(os.path.realpath(__file__))
    
    for entry in os.scandir(directory):
        if entry.path.endswith(".h5") and entry.is_file():
            if training_mode is None or \
               (training_mode == "ClimbLevel" and "ClimbLevel" in entry.name) or \
               (training_mode == "OneCliffLevel" and "OneCliffLevel" in entry.name) or \
               (training_mode == "RoughTerrainLevel" and "RoughTerrainLevel" in entry.name) or \
               (training_mode == "CliffsAndEnemiesLevel" and "CliffsAndEnemiesLevel" in entry.name):
                print(entry.path)
                if dataset is not None:
                    for episode in ReplayBuffer.load(entry.path, InfiniteBuffer()).episodes:
                        dataset.append_episode(episode)
                else:
                    dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
    return dataset