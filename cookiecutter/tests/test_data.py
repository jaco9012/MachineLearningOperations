import torch
import os
import pytest
from src.data.dataset import mnist



@pytest.mark.skipif(not os.path.exists("data/processed/test_images.pt"), reason="Test images not found")
@pytest.mark.skipif(not os.path.exists("data/processed/test_labels.pt"), reason="Test labels not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Train images not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train_labels.pt"), reason="Train labels not found")
class TestDataSet:
    """Class to perform test on the data set"""

    def test_length(self):
        self.train_set, self.test_set = mnist()
        self.trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=len(self.train_set), shuffle=False)
        self.testloader  = torch.utils.data.DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        self.train_dataset = next(iter(self.trainloader))
        self.test_dataset = next(iter(self.testloader))
        assert len(self.train_dataset[0]) == 40000 and len(self.test_dataset[0]) == 5000, "Dataset did not have the correct number of samples"
    
    def test_shape(self):
        self.train_set, self.test_set = mnist()
        self.trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=len(self.train_set), shuffle=False)
        self.testloader  = torch.utils.data.DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        self.train_dataset = next(iter(self.trainloader))
        self.test_dataset = next(iter(self.testloader))
        train_images, _ = self.train_dataset
        test_images, _  = self.test_dataset
        assert train_images.shape == (40000, 28, 28) and test_images.shape == (5000, 28, 28), "The test data set did not have the correct dimension"

    def test_labels(self):
        self.train_set, self.test_set = mnist()
        self.trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=len(self.train_set), shuffle=False)
        self.testloader  = torch.utils.data.DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        self.train_dataset = next(iter(self.trainloader))
        self.test_dataset = next(iter(self.testloader))
        _, train_labels = self.train_dataset
        _, test_labels = self.test_dataset
        assert all(i in train_labels for i in range(10)), "Not all labels are represented in the training data"

