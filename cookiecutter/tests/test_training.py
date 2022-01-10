import torch
from models.model import MyAwesomeModel


class TestTraining:
    """Class to perform test on the model"""
    def UniformInit(self):
        self.model = MyAwesomeModel()
        self.model.train()
        self.testImage = torch.rand([1, 28, 28])
        
        # the output should be 10 logprobabilities
        assert torch.all(0.1+0.05 >= torch.exp(self.model(self.testImage))) and torch.all(0.1-0.05 <= torch.exp(self.model(self.testImage))), "The model is not uniformely initialzed"
