import torch.nn as nn


class HardMixtureOfExpertsNetwork(nn.Module):
    def __init__(self, num_experts):
        super().__init__()

    def forward(self, x):
        return x
