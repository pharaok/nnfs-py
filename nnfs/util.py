import math


def dot(a: list[float], b: list[float]) -> float:
    """Compute the dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function."""
    return 1 / (1 + math.exp(-x))
