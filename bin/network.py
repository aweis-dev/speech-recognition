import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax

    def forward(self, input, hidden):
        combined= torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, hidden_size):
        return torch.zeros(1, hidden_size)

if __name__ == "__main__":
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
