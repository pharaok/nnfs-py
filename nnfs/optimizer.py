from dataclasses import dataclass
from nnfs.matrix import Matrix


@dataclass
class SGD:  # Stochastic Gradient Descent
    learning_rate: float
    current_learning_rate: float
    decay: float
    momentum: float
    iterations: int

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        # if self.momentum:
        #     if not hasattr(layer, "weight_momentums"):
        #         layer.weight_momentums = Matrix.empty_like(layer.weights)
        #         layer.bias_momentums = Matrix.empty_like(layer.biases)
        #
        #     weight_updates = (
        #         layer.weight_momentums * self.momentum
        #         - self.current_learning_rate * layer.dweights
        #     )
        #     layer.weight_momentums = weight_updates
        #     bias_updates = (
        #         layer.bias_momentums * self.momentum
        #         - self.current_learning_rate * layer.dbiases
        #     )
        #     layer.bias_momentums = bias_updates
        # else:
        weight_updates = -self.current_learning_rate * layer.dweights
        bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights = layer.weights + weight_updates
        layer.biases = layer.biases + bias_updates

    def post_update_params(self):
        self.iterations += 1
