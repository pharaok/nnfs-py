from nnfs.layer import Layer, ActivationFunction
from nnfs.matrix import Matrix

l1 = Layer.from_weights_and_biases(
    Matrix.from_rows(
        [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ]
    ).transposed(),
    [2.0, 3.0, 0.5],
)

inputs = Matrix.from_rows([[1.0, 2.0, 3.0, 2.5]])
print(l1)
print(l1.forward(inputs))
