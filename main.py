from nnfs.layer import Layer, ActivationFunction
from nnfs.neuron import Neuron
from nnfs.matrix import Matrix

l1 = Layer.from_weights_and_biases(
    [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ],
    [2.0, 3.0, 0.5],
    activation_function=ActivationFunction.SOFTMAX,
)

inputs = Matrix.from_col([1.0, 2.0, 3.0, 2.5])
print(l1)
print(l1.forward(inputs))
