import torch
from torch import nn, Tensor

class TestNet(nn.Module):
  def __init__(self, size=10):
    super(TestNet, self).__init__()
    self.fc = nn.Linear(size*size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 4)
  def forward(self, x):
    x = x.view(-1, 100)
    x = torch.relu(self.fc(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# mse loss
def loss(x:Tensor, y:Tensor) -> Tensor:
  return torch.mean((x-y)**2)