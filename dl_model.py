import torch
import torch.nn.functional as F
import torch.nn as nn
from data_pipeline import X_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_inputs = X_train.shape[1]
n_hidden1 = 200
n_hidden2 = 50
n_outputs = 1
dropout = 0.0

class Regression(nn.Module):
    def __init__(self, n_hidden1, n_hidden2, dropout_prob=dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden1 = nn.Linear(n_inputs, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_outputs)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

net = Regression(n_hidden1, n_hidden2)