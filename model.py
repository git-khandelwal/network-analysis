import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, output_size=4):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, 19)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x