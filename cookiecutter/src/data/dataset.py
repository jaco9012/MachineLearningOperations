import torch
import os
import numpy as np

class dataset(torch.utils.data.Dataset):
        """Corrupted MNIST training data"""

        def __init__(self, train):
            """
            Args:
                file (string)
            """

            if train == True:
                self.images = torch.load('data/processed/train_images.pt')
                self.labels = torch.load('data/processed/train_labels.pt')
            else:
                self.images = torch.load('data/processed/test_images.pt')
                self.labels = torch.load('data/processed/test_labels.pt')                

        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):

            images = self.images[idx]
            labels = self.labels[idx]

            return images, labels

def mnist():
    
    train_set = dataset(train=True)
    test_set = dataset(train=False)

    return train_set, test_set
