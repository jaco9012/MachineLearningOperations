import torch.nn.functional as F
from torch import nn

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
	if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
		nn.init.orthogonal_(module.weight.data, gain)
		nn.init.constant_(module.bias.data, 0)
	return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MyAwesomeModel(nn.Module):
    def __init__(self, in_channels=1, feature_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=5, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=4096, out_features=1024), nn.ReLU(),
            nn.Linear(in_features=1024, out_features=feature_dim) 
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        x = x[:,None,:,:]
        return F.log_softmax(self.layers(x), dim=1)