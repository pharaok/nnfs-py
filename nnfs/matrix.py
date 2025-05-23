from dataclasses import dataclass
from abc import ABC, abstractmethod
from nnfs.util import dot
import random


class DimensionError(Exception):
    """Custom exception for dimension errors in matrix operations."""

    pass


@dataclass
class Matrix:
    n_rows: int
    n_cols: int
    _rows: list[list[float]]

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._rows = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

    @classmethod
    def random(
        cls, n_rows: int, n_cols: int, low: float = -1.0, high: float = 1.0
    ) -> "Matrix":
        matrix = cls(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                matrix[i][j] = random.uniform(low, high)
        return matrix

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        matrix = cls(n, n)
        for i in range(n):
            matrix[i][i] = 1.0
        return matrix

    @classmethod
    def ones(cls, n_rows: int, n_cols: int) -> "Matrix":
        matrix = cls(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                matrix[i][j] = 1.0
        return matrix

    @classmethod
    def from_rows(cls, rows: list[list[float]]) -> "Matrix":
        if isinstance(rows[0], Matrix):
            rows = [rows[i]._rows[0] for i in range(len(rows))]
        matrix = cls(len(rows), len(rows[0]))
        matrix._rows = rows
        return matrix

    @classmethod
    def from_row(cls, row: list[float]) -> "Matrix":
        matrix = cls(1, len(row))
        matrix._rows = [row]
        return matrix

    @classmethod
    def from_col(cls, col: list[float]) -> "Matrix":
        matrix = cls(len(col), 1)
        matrix._rows = [[x] for x in col]
        return matrix

    @classmethod
    def empty_like(cls, m: "Matrix") -> "Matrix":
        return cls(m.n_rows, m.n_cols)

    def __getitem__(self, index: int) -> list[float]:
        return self._rows[index]

    def __setitem__(self, index: int, value: list[float]):
        self._rows[index] = value

    # def rows(self) -> list["Matrix"]:
    #     return [Matrix.from_row(row) for row in self._rows]
    #
    # def cols(self) -> list["Matrix"]:
    #     return [Matrix.from_col(col) for col in self.transposed()._rows]
    #
    # def row(self, index: int) -> "Matrix":
    #     return self.rows()[index]
    #
    # def col(self, index: int) -> "Matrix":
    #     return self.cols()[index]

    def transposed(self) -> "Matrix":
        transposed = Matrix(self.n_cols, self.n_rows)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                transposed[j][i] = self[i][j]
        return transposed

    def to_list(self) -> list[list[float]]:
        return self._rows

    def __neg__(self) -> "Matrix":
        return -1 * self

    def __add__(self, other: "Matrix") -> "Matrix":
        m, n, k, l = self.n_rows, self.n_cols, other.n_rows, other.n_cols
        if not (m % k == 0 and n % l == 0):
            raise DimensionError(
                f"Cannot add matrices with dimensions: {m}x{n} and {k}x{l}"
            )

        result = Matrix(self.n_rows, self.n_cols)
        for i in range(m):
            for j in range(n):
                result[i][j] = self[i][j] + other[i % k][j % l]
        return result

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self + (-other)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.n_cols != other.n_rows:
            raise DimensionError(
                f"""Cannot multiply matrices with dimensions: {self.n_rows}x{
                    self.n_cols} and {other.n_rows}x{other.n_cols}"""
            )
        result = Matrix(self.n_rows, other.n_cols)
        other = other.transposed()
        for i in range(self.n_rows):
            for j in range(other.n_rows):  # transposed
                result[i][j] = dot(self[i], other[j])
        return result

    def __mul__(self, a: float) -> "Matrix":
        result = self.copy()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                result[i][j] *= a
        return result

    def __rmul__(self, a: float) -> "Matrix":
        return self * a

    def __truediv__(self, a: float) -> "Matrix":
        return self * (1 / a)

    def copy(self) -> "Matrix":
        return Matrix.from_rows([row.copy() for row in self._rows])

    def map(self, func) -> "Matrix":
        result = Matrix(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                result[i][j] = func(self[i][j])
        return result

    def sum(self, axis: int = None) -> float or "Matrix":
        if axis is None:
            return sum(sum(row) for row in self._rows)
        if axis == 0:
            result = Matrix(1, self.n_cols)
            for j in range(self.n_cols):
                result[0][j] = sum(self[i][j] for i in range(self.n_rows))
            return result
        if axis == 1:
            result = Matrix(self.n_rows, 1)
            for i in range(self.n_rows):
                result[i][0] = sum(self[i])
            return result

    def mean(self, axis: int = None) -> float or "Matrix":
        if axis is None:
            return self.sum() / (self.n_rows * self.n_cols)
        if axis == 0:
            return self.sum(axis) / self.n_rows
        if axis == 1:
            return self.sum(axis) / self.n_cols

    def max(self, axis: int = None) -> float or "Matrix":
        if axis is None:
            return max(max(row) for row in self._rows)
        if axis == 0:
            result = Matrix(1, self.n_cols)
            for j in range(self.n_cols):
                result[0][j] = max(self[i][j] for i in range(self.n_rows))
            return result
        if axis == 1:
            result = Matrix(self.n_rows, 1)
            for i in range(self.n_rows):
                result[i][0] = max(self[i])
            return result

    def argmax(self, axis: int = 0) -> "Matrix":
        if axis == 0:
            return Matrix.from_row(
                [
                    max(range(self.n_rows), key=lambda i: self[i][j])
                    for j in range(self.n_cols)
                ]
            )
        if axis == 1:
            return Matrix.from_col(
                [
                    max(range(self.n_cols), key=lambda j: self[i][j])
                    for i in range(self.n_rows)
                ]
            )


class Differentiable(ABC):
    @abstractmethod
    def forward(cls, inputs: Matrix) -> Matrix:
        pass

    @abstractmethod
    def backward(cls, inputs: Matrix) -> Matrix:
        pass
