import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
        """Corrupted MNIST training data"""

        def __init__(self, file = "train_0.npz", root_dir = "/mnt/c/Users/jacob/OneDrive/Dokumenter/DTU/Machine Learning Operations/dtu_mlops/data/corruptmnist/", transform=None):
            """
            Args:
                file (string)
            """
            data = np.load(root_dir + file)
            images = torch.FloatTensor(data["images"])
            self.images = torch.flatten(images, start_dim=1)
            self.labels = torch.LongTensor(data["labels"])

        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):

            images = self.images[idx]
            labels = self.labels[idx]

            return images, labels

def mnist():
    
    train_set = dataset(file = "train_0.npz", root_dir = "/mnt/c/Users/jacob/OneDrive/Dokumenter/DTU/Machine Learning Operations/dtu_mlops/data/corruptmnist/")
    test_set = dataset(file = "test.npz", root_dir = "/mnt/c/Users/jacob/OneDrive/Dokumenter/DTU/Machine Learning Operations/dtu_mlops/data/corruptmnist/")

    return train_set, test_set
