from torch import nn


class BaseLine(nn.Module):
    """
    Simple 2-hidden layer non-linear neural network
    """

    def __init__(self, input_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 output_size):
        super(BaseLine, self).__init__()
        self.linear_layer_1 = nn.Linear(input_size, hidden_size_1)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size_1)
        self.activation_1 = nn.SELU()
        self.linear_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size_2)
        self.activation_2 = nn.SELU()
        self.linear_layer_3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.batch_norm_3 = nn.BatchNorm1d(hidden_size_3)
        self.activation_3 = nn.SELU()
        self.linear_layer_4 = nn.Linear(hidden_size_3, output_size)
        self.batch_norm_4 = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(p=0.5) # 0.3

    def forward(self, x):
        d_1 = self.dropout(
                self.activation_1(
                    self.batch_norm_1(
                        self.linear_layer_1(x))))
        d_2 = self.dropout(
            self.activation_2(
                self.batch_norm_2(
                    self.linear_layer_2(d_1))))
        d_3 = self.dropout(
                self.activation_3(
                    self.batch_norm_3(
                        self.linear_layer_3(d_2))))
        l_4 = self.batch_norm_4(self.linear_layer_4(d_3))
        return l_4
