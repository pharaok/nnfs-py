from nnfs.matrix import Matrix, Differentiable
from nnfs.loss import Crossentropy
import math


class ReLU(Differentiable):
    def forward(self, inputs: Matrix):
        self.inputs = inputs
        self.outputs = inputs.map(lambda x: max(0, x))

    def backward(self, dvalues: Matrix):
        self.dinputs = dvalues.copy()
        for i in range(self.dinputs.n_rows):
            for j in range(self.dinputs.n_cols):
                if self.inputs[i][j] <= 0:
                    self.dinputs[i][j] = 0


class Softmax(Differentiable):
    def forward(self, inputs: Matrix):
        self.inputs = inputs

        inputs = inputs - inputs.max(axis=1)
        exp = inputs.map(math.exp)
        for i in range(exp.n_rows):
            s = sum(exp._rows[i])
            for j in range(exp.n_cols):
                exp[i][j] /= s  # Normalize
        self.outputs = exp

    def backward(self, dvalues: Matrix):
        self.dinputs = Matrix.empty_like(dvalues)
        for i in range(dvalues.n_rows):
            output = self.outputs._rows[i]
            dvalue = dvalues._rows[i]
            nc = len(output)

            # Calculate the Jacobian matrix
            jacobian = Matrix(nc, nc)
            for j in range(nc):
                for k in range(nc):
                    if j == k:
                        jacobian[j][k] = output[j] * (1 - output[j])
                    else:
                        jacobian[j][k] = -output[j] * output[k]

            # Calculate sample gradient
            sample_dinputs = jacobian @ Matrix.from_col(dvalue)

            for j in range(dvalues.n_cols):
                self.dinputs[i][j] = sample_dinputs[j][0]


class Softmax_Crossentropy(Differentiable):
    def __init__(self):
        self.activation = Softmax()
        self.loss = Crossentropy()

    def forward(self, inputs: Matrix, y_true: Matrix):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues: Matrix, y_true: Matrix):
        samples = dvalues.n_rows
        self.dinputs = dvalues.copy()
        self.dinputs = self.dinputs - y_true
        self.dinputs = self.dinputs / samples
