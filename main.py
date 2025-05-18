from nnfs.layer import Layer
from nnfs.neuron import Neuron

l1 = Layer(3, 4)
l1.neurons[0] = Neuron([0.2, 0.8, -0.5, 1.0], 2.0)
l1.neurons[1] = Neuron([0.5, -0.91, 0.26, -0.5], 3.0)
l1.neurons[2] = Neuron([-0.26, -0.27, 0.17, 0.87], 0.5)

inputs = [1.0, 2.0, 3.0, 2.5]

print(l1)

print(l1.forward(inputs))
