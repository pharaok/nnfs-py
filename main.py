from nnfs.layer import Layer
from nnfs.matrix import Matrix
from nnfs.activation import Softmax, Softmax_Crossentropy
from nnfs.loss import Crossentropy
from nnfs.data import read_images, read_labels
from nnfs.optimizer import SGD
import timeit
import cProfile
import pstats

# import numpy as np

import matplotlib.pyplot as plt


# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
# def create_data(samples, classes):
#     X = np.zeros((samples * classes, 2))
#     y = np.zeros(samples * classes, dtype="uint8")
#     for class_number in range(classes):
#         ix = range(samples * class_number, samples * (class_number + 1))
#         r = np.linspace(0.0, 1, samples)
#         t = (
#             np.linspace(class_number * 4, (class_number + 1) * 4, samples)
#             + np.random.randn(samples) * 0.2
#         )
#         X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
#         y[ix] = class_number
#     return X, y


# m1 = Matrix.from_rows([[1, 2, 3], [4, 6, 3], [9, 9, 7]])
# tr = Matrix.from_rows([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
# s = Softmax()
# ll = Crossentropy()
#
# s.forward(m1)
# print(s.outputs)
#
# m1 = m1 - m1.max(axis=1)
# s.forward(m1)
# print(s.outputs)
#
# print(ll.calculate(s.outputs, tr))
# ll.backward(s.outputs, tr)
# print("1,", s.outputs)
# print("1,", ll.dinputs)
#
# sl = Softmax_Crossentropy()
#
# print(sl.forward(m1, tr))
#
# sl.backward(sl.outputs, tr)
# print("3,", sl.outputs)
# print("3,", sl.dinputs)
#
#
# exit()

# N = 20
# X, y = create_data(N, 3)
# plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="viridis")
# plt.show()

# mX = Matrix.from_rows(X).map(float)
# mY = Matrix(mX.n_rows, 3).map(float)
# for i in range(mX.n_rows):  # one-shot encoding
#     mY[i][y[i]] = 1.0
# print(mX.n_rows, mX.n_cols)
# print(mY.n_rows, mY.n_cols)


dense1 = Layer(784, 256)
dense2 = Layer(256, 64)
dense3 = Layer(64, 10, activation_function=Softmax_Crossentropy)
optimizer = SGD(0.01, decay=1e-6, momentum=0.9)

# cProfile.run(
#     """
# for _ in range(1000):
#     dense1.forward(mX)
#     dense2.forward(dense1.outputs, mY)
#     dense2.backward(dense2.outputs, mY)
#     dense1.backward(dense2.dinputs)
# """,
#     sort="tottime",
# )
# exit(0)


x_test = read_images("data/train-images-idx3-ubyte")
y_test = read_labels("data/train-labels-idx1-ubyte")
print("Loaded dataset")
# print(x_train)
# print(y_train)
# exit(0)

for epoch in range(1):
    batch_size = 100
    batch_start = epoch * batch_size
    x_batch = Matrix.from_rows(x_test[batch_start : batch_start + batch_size]).map(
        float
    )
    y_batch = Matrix(batch_size, 10)
    for i in range(batch_size):
        y_batch[i][y_test[batch_start + i]] = 1.0

    dense1.forward(x_batch)
    dense2.forward(dense1.outputs)
    loss = dense3.forward(dense2.outputs, y_batch)

    dense3.backward(dense3.outputs, y_batch)
    dense2.backward(dense3.dinputs)
    dense1.backward(dense2.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

    if (epoch % 1) == 0:
        print(f"Iteration {epoch}: " f"Loss: {loss} ")

    # print(dense1.dweights)
    # print(dense1.dbiases)
    # print(dense2.dweights)
    # print(dense2.dbiases)

print("Training complete")

x_test = read_images("data/train-images-idx3-ubyte")
y_test = read_labels("data/train-labels-idx1-ubyte")
print("Loaded dataset")

current_index = 0


def guess():
    global current_index
    i = current_index
    dense1.forward(Matrix.from_row(x_test[i]).map(float))
    dense2.forward(dense1.outputs)
    dense3.forward(dense2.outputs, Matrix(1, 10).map(float))
    guess_label = dense3.outputs.argmax(axis=1)[i]
    return guess_label


def plot_image(index):
    plt.clf()
    image = [[0] * 28 for _ in range(28)]
    for i in range(28):
        for j in range(28):
            image[i][j] = x_test[index][i * 28 + j] / 255.0

    plt.imshow(image, cmap="gray")

    plt.title(f"Image {index+1}/{len(x_test)}: {y_test[index]} (guessed: {guess()})")
    plt.axis("off")
    plt.show()


def on_key(event):
    global current_index
    if event.key in ["right", " "]:
        current_index = (current_index + 1) % len(x_test)
        plot_image(current_index)
    elif event.key in ["left", "backspace"]:
        current_index = (current_index - 1) % len(x_test)
        plot_image(current_index)
    if event.key in ["q", "escape"]:
        plt.close()


print(plt.get_backend())
fig = plt.figure()
plot_image(current_index)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
