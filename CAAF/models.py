"""Small regression models accepting flat inputs or temporal windows."""

import math

from torch import nn


# Construct a new activation at each layer so modules are never shared.
def _activation(name, negative_slope=0.01):
    choices = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(negative_slope),
               "gelu": nn.GELU, "tanh": nn.Tanh}
    if name not in choices:
        raise ValueError(f"Unknown activation {name!r}; choose {tuple(choices)}.")
    return choices[name]()


# The MLP flattens all history steps before its hidden layers.
class MLP(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_sizes=(64, 64, 64),
                 activation="leaky_relu", negative_slope=0.01, dropout=0,
                 batch_norm=False, bias=True):
        super().__init__()
        layers = [nn.Flatten()]
        width = math.prod(input_shape)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(width, hidden, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(_activation(activation, negative_slope))
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, output_dim, bias=bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# The LSTM combines the final hidden states from the last recurrent layer.
class LSTM(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_size=64, num_layers=2,
                 dropout=0, bidirectional=False):
        super().__init__()
        self.recurrent = nn.LSTM(input_shape[-1], hidden_size, num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                 bidirectional=bidirectional)
        self.output = nn.Linear(hidden_size * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        _, (hidden, _) = self.recurrent(x)
        summary = hidden[-2:].transpose(0, 1).flatten(1) if self.recurrent.bidirectional else hidden[-1]
        return self.output(summary)


# Left padding makes each temporal convolution causal and preserves history length.
class TCN(nn.Module):
    def __init__(self, input_shape, output_dim, channels=(64, 64), kernel_size=3,
                 activation="relu", dropout=0):
        super().__init__()
        layers = []
        width = input_shape[-1]
        for channel in channels:
            layers.extend([nn.ConstantPad1d((kernel_size - 1, 0), 0),
                           nn.Conv1d(width, channel, kernel_size),
                           _activation(activation)])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = channel
        self.temporal = nn.Sequential(*layers)
        length = input_shape[0] if len(input_shape) == 2 else 1
        self.output = nn.Linear(width * length, output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.output(self.temporal(x.transpose(1, 2)).flatten(1))
