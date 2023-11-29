import torch.nn.functional as F
from torch import nn
from typing import List, Union, Callable
import torch


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
        use_batch_norm: bool = False,  # New argument for batch normalization
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
            if use_batch_norm:  # Add batch normalization layer if use_batch_norm is True
                self.layers.append(nn.BatchNorm1d(layer_dim))
            self.layers.append(act_fn)
            self.layers.append(nn.Dropout(dropout_rate))
            prev_layer_dim = layer_dim

        # Create the output layer
        self.layers.append(nn.Linear(prev_layer_dim, output_dim))
        self.layers.append(output_activation)

        if use_double_precision:
            self.double()  # Convert to double precision

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvexNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        """
        Initialize the modified neural network.

        :param input_dim: Number of input dimensions.
        :param output_dim: Number of output dimensions.
        :param hidden_layers: List containing the number of neurons in each hidden layer.
        """
        super(ConvexNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers.append(output_dim)

        # Main layers from previous hidden layer
        self.main_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            linear = nn.Linear(prev_dim, hidden_dim)
            if prev_dim != input_dim:  # Apply positivity constraint only to layers after the first
                with torch.no_grad():
                    linear.weight.clamp_(min=0)
            self.main_layers.append(linear)
            prev_dim = hidden_dim

        # Additional layers from the original input
        self.additional_layers = nn.ModuleList()
        for hidden_dim in hidden_layers[1:]:  # Skip first layer as it directly connects to input
            linear = nn.Linear(input_dim, hidden_dim)
            with torch.no_grad():
                linear.weight.clamp_(min=0)
            self.additional_layers.append(linear)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for i, main_layer in enumerate(self.main_layers):
            x_main = main_layer(x if i == 0 else output)

            # Apply additional layer if not the first hidden layer
            if i > 0:
                x_additional = self.additional_layers[i-1](x)
                output = nn.ReLU()(x_main + x_additional)
            if i == len(self.main_layers) - 1:
                output = x_main
            else:
                output = nn.ReLU()(x_main)

        return output
    
    def clamp_weights(self):
        """
        Ensure that the weights remain positive after training updates, except for the first layer.
        """
        for i, layer in enumerate(self.main_layers):
            if i > 0 and isinstance(layer, nn.Linear):  # Skip the first layer
                with torch.no_grad():
                    layer.weight.clamp_(min=0)


class NeuralReach(MLP):
    def __init__(self, hidden_layers=[32, 64, 32]):
        input_dim = 5  # alt, vx, vz, z, tgo
        output_dim = 4  # xmin, xmax, ymax, x-ymax
        activation_fn = nn.ReLU()
        output_activation = nn.Sigmoid()
        dropout_rate = 0.0
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

class NeuralReachCvx(ConvexNN):
    def __init__(self, hidden_layers=[32, 64, 32]):
        input_dim = 5  # alt, vx, vz, z, tgo
        output_dim = 4  # xmin, xmax, ymax, x-ymax
        super(NeuralReachCvx, self).__init__(
            input_dim,
            output_dim,
            hidden_layers,
        )

    def forward(self, x):
        out = super(NeuralReachCvx, self).forward(x)
        # change sign of second (xmax) and third (ymax) output to ensure convexity
        out[:, 1] = -out[:, 1]
        out[:, 2] = -out[:, 2]
        return out


class NeuralFeasibility(MLP):
    def __init__(self, hidden_layers):
        input_dim = 5  # alt, vx, vz, z, tgo
        output_dim = 1  # feasibility
        activation_fn = nn.ReLU()
        output_activation = nn.Sigmoid()
        dropout_rate = 0.0
        use_double_precision = True
        use_batch_norm = True
        super(NeuralFeasibility, self).__init__(
            input_dim,
            output_dim,
            hidden_layers,
            activation_fn,
            output_activation,
            dropout_rate,
            use_double_precision,
            use_batch_norm,
        )


class NeuralFeasibilityCvx(ConvexNN):
    def __init__(self, hidden_layers):
        input_dim = 5  # alt, vx, vz, z, tgo
        output_dim = 1  # feasibility
        super(NeuralFeasibilityCvx, self).__init__(
            input_dim,
            output_dim,
            hidden_layers,
        )

    def forward(self, x):
        out = super(NeuralFeasibilityCvx, self).forward(x)
        return -out  # change sign to ensure concavity