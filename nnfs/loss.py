from nnfs.util import clamp, dot
from nnfs.matrix import Matrix
import math


class Loss:
    def calculate(self, output: Matrix, y_true: Matrix) -> float:
        losses = self.forward(output, y_true)
        return losses.mean()


class Crossentropy(Loss):
    def forward(self, inputs: Matrix, y_true: Matrix):
        inputs = inputs.map(lambda x: clamp(x, 1e-7, 1 - 1e-7))  # Clip

        self.outputs = Matrix(inputs.n_rows, 1)
        for i in range(inputs.n_rows):
            self.outputs[i][0] = -math.log(
                max(1e-7, dot(inputs._rows[i], y_true._rows[i]))
            )
        return self.outputs

    def backward(self, dvalues: Matrix, y_true: Matrix):
        samples = dvalues.n_rows

        self.dinputs = Matrix.empty_like(dvalues)
        for i in range(dvalues.n_rows):
            for j in range(dvalues.n_cols):
                self.dinputs[i][j] = -y_true[i][j] / dvalues[i][j]
        self.dinputs = self.dinputs / samples  # Normalize
