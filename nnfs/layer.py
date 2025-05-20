from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass
from nnfs.matrix import Matrix, DimensionError
from nnfs.util import sigmoid, softmax


class ActivationFunction(Enum):
    NONE = "none"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


@dataclass
class Layer:
    # neurons: list[Neuron]
    weights: Matrix  # n_inputs x n_neurons
    biases: Matrix  # 1 x n_neurons
    _activation_function: Callable[[list[float]], list[float]]

    @property
    def activation_function(self) -> Callable[[list[float]], list[float]]:
        return self._activation_function

    @activation_function.setter
    def activation_function(self, af: Callable[[list[float]], list[float]]):
        if af == ActivationFunction.NONE:
            self._activation_function = lambda xs: xs
        if af == ActivationFunction.SIGMOID:
            self._activation_function = lambda xs: [sigmoid(x) for x in xs]
        if af == ActivationFunction.SOFTMAX:
            self._activation_function = softmax

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        activation_function=ActivationFunction.NONE,
    ):
        self.weights = Matrix(n_neurons, n_inputs)
        self.biases = Matrix(n_neurons, 1)
        self.activation_function = activation_function

    @classmethod
    def from_weights_and_biases(
        cls,
        weights: Matrix,
        biases: Matrix,
        activation_function=ActivationFunction.NONE,
    ):
        if not isinstance(weights, Matrix):
            weights = Matrix.from_rows(weights)
        if not isinstance(biases, Matrix):
            biases = Matrix.from_row(biases)

        if weights.n_cols != biases.n_cols or biases.n_rows != 1:
            raise DimensionError
        layer = cls(weights.n_rows, weights.n_cols)
        layer.weights = weights
        layer.biases = biases
        layer.activation_function = activation_function
        return layer

    def forward(self, inputs: Matrix) -> Matrix:
        return Matrix.from_rows(
            [
                self._activation_function((r + self.biases)._rows[0])
                for r in (inputs * self.weights).rows()
            ]
        )
