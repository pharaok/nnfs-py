from dataclasses import dataclass


@dataclass
class SGD:  # Stochastic Gradient Descent
    learning_rate: float
    current_learning_rate: float
    decay: float
    iterations: int

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        weight_updates = -self.current_learning_rate * layer.dweights
        bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights = layer.weights + weight_updates
        layer.biases = layer.biases + bias_updates

    def post_update_params(self):
        self.iterations += 1
