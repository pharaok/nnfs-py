import math


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def softmax(xs: list[float]) -> list[float]:
    exp_xs = [math.exp(x) for x in xs]
    sum_exp_xs = sum(exp_xs)
    return [x / sum_exp_xs for x in exp_xs]
