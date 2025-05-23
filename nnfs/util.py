def dot(a: list[float], b: list[float]) -> float:
    # this was faster than sum(x * y for x, y in zip(a, b))
    # at least on pypy3
    dot_prod = 0
    for x, y in zip(a, b):
        dot_prod += x * y
    return dot_prod


def clamp(x: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(x, max_value))


def clip(x: float) -> float:
    return clamp(x, 1e-7, 1 - 1e-7)
