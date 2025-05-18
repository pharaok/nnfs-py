from dataclasses import dataclass
from nnfs.util import dot


@dataclass
class Neuron:
    """
    A single neuron in a neural network.

    A forward pass through the neuron is the dot product of the inputs and
    weights, plus the bias, passed through an activation function.
    """

    weights: list[float]
    bias: float
    activation_function: callable

    def __init__(
        self,
        weights: int | list[float],
        bias=0.0,
        activation_function=lambda x: x,
    ):
        if isinstance(weights, int):
            self.weights = [0.0] * weights
        else:
            self.weights = weights

        self.bias = bias
        self.activation_function = activation_function

    def forward(self, inputs: list[float]) -> float:
        return self.activation_function(dot(inputs, self.weights) + self.bias)
