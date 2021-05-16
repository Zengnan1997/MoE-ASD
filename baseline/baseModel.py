import torch
from torch import nn
IN_DIM = 600
N_HIDDEN_1 = 400
N_HIDDEN_2 = 200
OUT_DIM = 1


class baseModel(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, out_dim=OUT_DIM):
        super(baseModel, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x, y):
        x_y = torch.cat((x, y), 1)
        x_y = nn.functional.relu(self.layer1(x_y))
        x_y = nn.functional.relu(self.layer2(x_y))
        x_y = self.layer3(x_y)

        x_y = torch.squeeze(x_y)
        return x_y

    def loss(self, x_y, label):
        criterion = nn.BCEWithLogitsLoss()

        return criterion(x_y, label)

