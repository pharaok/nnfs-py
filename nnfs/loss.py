from nnfs.util import clamp
from nnfs.matrix import Matrix
import math


class Loss:
    def calculate(self, output: Matrix, y_true: Matrix) -> float:
        return self.forward(output, y_true).mean()


class Crossentropy(Loss):
    def forward(self, inputs: Matrix, y_true: Matrix):
        inputs = inputs.map(lambda x: clamp(x, 1e-7, 1 - 1e-7))  # Clip
        self.inputs = inputs
        self.outputs = inputs @ y_true.transposed()
        self.outputs = self.outputs.map(lambda x: -math.log(x))
        return self.outputs

    def backward(self, dvalues: Matrix, y_true: Matrix):
        samples = dvalues.n_rows

        self.dinputs = Matrix.empty_like(dvalues)
        for i in range(dvalues.n_rows):
            for j in range(dvalues.n_cols):
                self.dinputs[i][j] = -y_true[i][j] / dvalues[i][j]
        self.dinputs = self.dinputs / samples  # Normalize
