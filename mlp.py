from torch import nn


def _which_activation(activation):
    if activation is None:
        return None
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "leaky-relu":
        return nn.LeakyReLU()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError(activation)


class MLP(nn.Module):
    def __init__(self, dims, dropout=None, activation="relu", batchnorm=False,
                 final_activation=None):
        super(MLP, self).__init__()
        if dropout is None:
            dropout = 1.
        assert 0. < dropout <= 1.
        layers = []
        activation = _which_activation(activation)
        final_activation = _which_activation(final_activation)
        for d in range(len(dims) - 1):
            layer = []
            layer.append(nn.Linear(dims[d], dims[d + 1]))
            if batchnorm:
                layer.append(nn.BatchNorm1d(dims[d + 1]))
            if d < len(dims) - 2 and activation is not None:
                layer.append(activation)
                if dropout < 1.:
                    layer.append(nn.Dropout(dropout))
            if d == len(dims) - 2 and final_activation is not None:
                layer.append(final_activation)
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
