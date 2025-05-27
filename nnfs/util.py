def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def clamp(x: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(x, max_value))


def clip(x: float) -> float:
    return clamp(x, 1e-7, 1 - 1e-7)
