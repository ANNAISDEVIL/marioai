import os
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer


def getDataset():
    dataset = None
    directory = os.path.dirname(os.path.realpath(__file__))
    for entry in os.scandir(directory):
        if entry.path.endswith(".h5") and entry.is_file():
            print(entry.path)
            if dataset is not None:
                dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
                # dataset.append(ReplayBuffer.load(entry.path, InfiniteBuffer()))
            else:
                dataset = ReplayBuffer.load(entry.path, InfiniteBuffer())
    return dataset
