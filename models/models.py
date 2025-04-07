from torch import nn

class Mine_MLP(nn.Module):

    def __init__(self, nb_hidden):
        super(Mine_MLP, self).__init__()
        self.fc1 = nn.Linear(2840, nb_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
