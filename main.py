from nnfs.layer import Layer
from nnfs.matrix import Matrix
from nnfs.activation import Softmax, Softmax_Crossentropy
from nnfs.loss import Crossentropy
from nnfs.optimizer import SGD
import timeit
import cProfile
import pstats

import numpy as np
import matplotlib.pyplot as plt


# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def create_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# l1 = Layer.from_weights_and_biases(
#     Matrix.from_rows(
#         [
#             [0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87],
#         ]
#     ).transposed(),
#     [2.0, 3.0, 0.5],
# )
# l_output = Layer.from_weights_and_biases(
#     Matrix.from_rows(
#         [
#             [0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13],
#         ]
#     ).transposed(),
#     [-1, 2, -0.5],
#     activation_function=Softmax,
# )

N = 30
X, y = create_data(N, 3)
# plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="viridis")
# plt.show()

mX = Matrix.from_rows(X)
mY = Matrix(mX.n_rows, 3)
for i in range(mX.n_rows):  # one-shot encoding
    mY[i][y[i]] = 1.0


dense1 = Layer(2, 64)
dense2 = Layer(64, 3, activation_function=Softmax_Crossentropy)
optimizer = SGD()

# for t in range(10001):
# cProfile.run(
#     """
# for _ in range(100):
#     dense1.forward(mX)
#     dense2.forward(dense1.outputs)""",
# )
# exit(0)

for t in range(10001):
    dense1.forward(mX)
    loss = dense2.forward(dense1.outputs, mY)

    dense2.backward(dense2.outputs, mY)
    dense1.backward(dense2.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    # if (t % 50) == 0:
    print(f"Iteration {t}: " f"Loss: {loss} ")

    # print(dense1.dweights)
    # print(dense1.dbiases)
    # print(dense2.dweights)
    # print(dense2.dbiases)
