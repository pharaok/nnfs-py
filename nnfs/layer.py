from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass
from nnfs.matrix import Matrix, DimensionError, Differentiable
from nnfs.activation import ReLU, Softmax


@dataclass
class Layer:
    # neurons: list[Neuron]
    weights: Matrix  # n_inputs x n_neurons
    biases: Matrix  # 1 x n_neurons
    activation_function: Differentiable

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        activation_function=ReLU,
    ):
        self.weights = Matrix.random(n_inputs, n_neurons) * 0.01
        self.biases = Matrix(1, n_neurons)
        self.activation_function = activation_function()

    # @classmethod
    # def from_weights_and_biases(
    #     cls,
    #     weights: Matrix,
    #     biases: Matrix,
    #     activation_function=ReLU,
    # ):
    #     if not isinstance(weights, Matrix):
    #         weights = Matrix.from_rows(weights)
    #     if not isinstance(biases, Matrix):
    #         biases = Matrix.from_row(biases)
    #
    #     if weights.n_cols != biases.n_cols or biases.n_rows != 1:
    #         raise DimensionError
    #     layer = cls(weights.n_rows, weights.n_cols, activation_function)
    #     layer.weights = weights
    #     layer.biases = biases
    #     return layer

    def forward(self, inputs: Matrix, *args) -> Matrix:
        self.inputs = inputs
        z = inputs @ self.weights + self.biases
        tmp = self.activation_function.forward(z, *args)
        self.outputs = self.activation_function.outputs
        return tmp

    def backward(self, dvalues: Matrix, *args) -> Matrix:
        self.activation_function.backward(dvalues, *args)
        dvalues = self.activation_function.dinputs

        self.dinputs = dvalues @ self.weights.transposed()
        self.dweights = self.inputs.transposed() @ dvalues
        self.dbiases = dvalues.sum(axis=0)
