import torch.nn.functional as F
from torch import nn
from typing import List, Union, Callable


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron model implemented in PyTorch.

    This model is a simple feedforward neural network with the flexibility to specify
    the number and sizes of hidden layers, activation functions, and a dropout rate.
    It optionally uses double precision for computations.

    Args:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_layers (List[int]): A list of integers specifying the number of units in each hidden layer.
        activation_fn (Union[List[Callable], Callable]): The activation function(s) to be used in the hidden layers.
            If a single function is provided, it will be used for all layers. If a list is provided, it should be the same length as hidden_layers.
        output_activation (Callable): The activation function for the output layer.
        dropout_rate (float): The dropout rate for regularization.
        use_double_precision (bool): If True, uses double precision for computations.

    Attributes:
        layers (torch.nn.ModuleList): The layers of the MLP.

    Example:
        >>> model = MLP(5, 1, [10, 20], activation_fn=[nn.ReLU(), nn.Tanh()], output_activation=nn.Identity(), dropout_rate=0.5)
        >>> print(model)

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation_fn: Union[List[Callable], Callable] = nn.ReLU(),
        output_activation: Callable = nn.Identity(),
        dropout_rate: float = 0.5,
        use_double_precision: bool = False,
    ):
        super(MLP, self).__init__()

        if isinstance(activation_fn, Callable):
            activation_fn = [activation_fn] * len(hidden_layers)

        assert len(activation_fn) == len(
            hidden_layers
        ), "If activation_fn is a list, it must be the same length as hidden_layers."

        self.layers = nn.ModuleList()

        # Create the hidden layers
        prev_layer_dim = input_dim
        for layer_dim, act_fn in zip(hidden_layers, activation_fn):
            self.layers.append(nn.Linear(prev_layer_dim, layer_dim))
            self.layers.append(act_fn)
            self.layers.append(nn.Dropout(dropout_rate))
            prev_layer_dim = layer_dim

        # Create the output layer
        self.layers.append(nn.Linear(prev_layer_dim, output_dim))
        self.layers.append(output_activation)

        if use_double_precision:
            self.double()  # Convert to double precision

    def forward(self, x):
        """
        Forward propagation of the MLP.

        Args:
            x (torch.Tensor): The input to the neural network.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralReach(MLP):
    def __init__(self):
        input_dim = 5  # alt, vx, vz, z, tgo
        output_dim = 7  # feasible, xmin, xmax, alpha, yp, a1, a2
        hidden_layers = [32, 64, 32]
        activation_fn = nn.ReLU()
        output_activation = nn.Sigmoid()
        dropout_rate = 0.5
        use_double_precision = True
        super(NeuralReach, self).__init__(
            input_dim,
            output_dim,
            hidden_layers,
            activation_fn,
            output_activation,
            dropout_rate,
            use_double_precision,
        )
