from nnfs.layer import Layer
from nnfs.matrix import Matrix
from nnfs.activation import Softmax_Crossentropy
from nnfs.data import read_images, read_labels
from nnfs.optimizer import SGD
import pickle

import matplotlib.pyplot as plt

# Set to True to load from pickle files, False to train from scratch
LOAD_FROM_PICKLE = True


if not LOAD_FROM_PICKLE:
    dense1 = Layer(784, 256)
    dense2 = Layer(256, 64)
    dense3 = Layer(64, 10, activation_function=Softmax_Crossentropy)
    optimizer = SGD(0.01, decay=1e-6)

    x_test = read_images("data/train-images-idx3-ubyte")
    y_test = read_labels("data/train-labels-idx1-ubyte")
    print("Loaded training dataset")

    batch_size = 100
    for epoch in range(60000 // batch_size):
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
            print(f"Epoch {epoch}: Loss: {loss} ")

    print("Training complete")

    with open("dense1.pkl", "wb") as f:
        pickle.dump(dense1, f)
    with open("dense2.pkl", "wb") as f:
        pickle.dump(dense2, f)
    with open("dense3.pkl", "wb") as f:
        pickle.dump(dense3, f)
else:
    with open("dense1.pkl", "rb") as f:
        dense1 = pickle.load(f)
    with open("dense2.pkl", "rb") as f:
        dense2 = pickle.load(f)
    with open("dense3.pkl", "rb") as f:
        dense3 = pickle.load(f)

    print("Loaded pretrained model")


x_test = read_images("data/t10k-images-idx3-ubyte")
y_test = read_labels("data/t10k-labels-idx1-ubyte")
print("Loaded testing dataset")

x = Matrix.from_rows(x_test).map(float)

y = Matrix(x.n_rows, 10)
for i in range(x.n_rows):
    y[i][y_test[i]] = 1.0

dense1.forward(x)
dense2.forward(dense1.outputs)
dense3.forward(dense2.outputs, y)
print("Testing complete")

guesses = dense3.outputs.argmax(axis=1)
correct = 0
for i in range(x.n_rows):
    if guesses[i][0] == y_test[i]:
        correct += 1
accuracy = correct / x.n_rows
print(f"Test accuracy: {accuracy}")
print(f"Correct guesses: {correct} out of {x.n_rows}")


current_index = 0


def plot_image(idx):
    plt.clf()
    image = [[0] * 28 for _ in range(28)]
    for i in range(28):
        for j in range(28):
            image[i][j] = x_test[idx][i * 28 + j] / 255.0

    plt.imshow(image, cmap="gray")

    g = guesses[idx][0]
    plt.title(f"Image {idx+1}/{len(x_test)}: {y_test[idx]} (guessed: {g})")
    plt.axis("off")
    plt.draw()


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


fig = plt.gcf()
plot_image(current_index)
cid = fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
