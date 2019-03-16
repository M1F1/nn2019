from torch import nn


class BaseLine(nn.Module):
    """
    Simple 2-hidden layer non-linear neural network
    """

    def __init__(self, input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size):
        super(BaseLine, self).__init__()
        self.linear_layer_1 = nn.Linear(input_size, hidden_size_1)
        self.activation_1 = nn.LeakyReLU(0.1)
        self.linear_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.activation_2 = nn.LeakyReLU(0.1)
        self.linear_layer_3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        l_1 = self.linear_layer_1(x)
        a_1 = self.activation_1(l_1)
        l_2 = self.linear_layer_2(a_1)
        a_2 = self.activation_2(l_2)
        return a_2
