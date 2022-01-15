import torch
import pytest
from models.model import MyAwesomeModel


class TestModel:
    """Class to perform test on the model"""
    def test_model_shape(self):
        self.model = MyAwesomeModel()
        self.model.train()
        self.testImage = torch.rand(1, 28, 28)
        # the output should be 10 logprobabilities
        assert self.model(self.testImage).shape[1] == 10, "The model does not output 10 predictions"

    def test_error_on_wrong_shape(self):
        with pytest.raises(ValueError, match='Expected input to a 3D tensor'):
            self.model = MyAwesomeModel()
            self.model(torch.randn(1,2))