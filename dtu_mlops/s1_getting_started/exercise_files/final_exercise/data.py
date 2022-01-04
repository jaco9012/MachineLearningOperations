from os import getcwd
import torch
import numpy as np


def mnist(trainingset):
    # exchange with the corrupted mnist dataset
    train = np.load("dtu_mlops/data/corruptmnist/" + trainingset + ".npz")
    train["images"] = torch.flatten(torch.from_numpy(train["images"]), start_dim=1)
    test = np.load("dtu_mlops/data/corruptmnist/test.npz")
    return train, test

