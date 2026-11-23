"""Small joint-outcome MLP. Its inputs are manufacturing observations only."""
import torch
from torch import nn

class RiskMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(34,32),nn.GELU(),nn.Dropout(.25),
            nn.Linear(32,16),nn.GELU(),nn.Dropout(.2),nn.Linear(16,8))
    def forward(self,x):return self.net(x)
