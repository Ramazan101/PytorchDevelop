import torch.nn as nn

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

class CheckCloths(nn.Module):
  def __init__(self):
    super().__init__()
    self.next = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.today = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16*14*14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
  def forward(self, x):
    x = self.next(x)
    x = self.today(x)
    return x


