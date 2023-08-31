import torch
from torch import nn
import torch.nn.functional as F


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96,
                            kernel_size=11, stride=4, padding=1)

        self.c2 = nn.Conv2d(in_channels=96, out_channels=256,
                            kernel_size=5, stride=1, padding=2)

        self.c3 = nn.Conv2d(in_channels=256, out_channels=384,
                            kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384,
                            kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256,
                            kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        self.f1 = nn.Linear(in_features=6400, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=1000)
        self.f4 = nn.Linear(in_features=1000, out_features=2)

    def forward(self, X):
        X = self.relu(self.c1(X))
        X = self.pool(X)
        X = self.relu(self.c2(X))
        X = self.pool(X)
        X = self.relu(self.c3(X))
        X = self.relu(self.c4(X))
        X = self.relu(self.c5(X))
        X = self.pool(X)

        X = self.flatten(X)
        X = self.relu(self.f1(X))
        X = F.dropout(X, 0.5)
        X = self.relu(self.f2(X))
        X = F.dropout(X, 0.5)
        X = self.relu(self.f3(X))
        X = F.dropout(X, 0.5)
        X = self.f4(X)
        return X


if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)
