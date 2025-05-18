from dataclasses import dataclass
from nnfs.neuron import Neuron


@dataclass
class Layer:
    neurons: list[Neuron]

    def __init__(self, n_neurons: int, n_inputs: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs: list[float]) -> list[float]:
        return [neuron.forward(inputs) for neuron in self.neurons]
