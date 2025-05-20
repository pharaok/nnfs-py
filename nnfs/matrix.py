from nnfs.util import dot
from dataclasses import dataclass


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
    def from_rows(cls, rows: list[list[float]]):
        matrix = cls(len(rows), len(rows[0]))
        matrix._rows = rows
        return matrix

    @classmethod
    def from_row(cls, row: list[float]):
        matrix = cls(1, len(row))
        matrix._rows = [row]
        return matrix

    @classmethod
    def from_col(cls, col: list[float]):
        matrix = cls(len(col), 1)
        matrix._rows = [[x] for x in col]
        return matrix

    def __getitem__(self, index: int) -> list[float]:
        return self._rows[index]

    def __setitem__(self, index: int, value: list[float]):
        self._rows[index] = value

    def rows(self) -> list["Matrix"]:
        return [Matrix.from_row(row) for row in self._rows]

    def cols(self) -> list["Matrix"]:
        return [Matrix.from_col(col) for col in self.transposed()._rows]

    def row(self, index: int) -> "Matrix":
        return self.rows()[index]

    def col(self, index: int) -> "Matrix":
        return self.cols()[index]

    def transposed(self) -> "Matrix":
        transposed = Matrix(self.n_cols, self.n_rows)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                transposed[j][i] = self[i][j]
        return transposed

    def to_list(self) -> list[list[float]]:
        return self._rows

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise DimensionError
        result = Matrix(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                result[i][j] = self[i][j] + other[i][j]
        return result

    def __mul__(self, other: "Matrix") -> "Matrix":
        if self.n_cols != other.n_rows:
            raise DimensionError
        result = Matrix(self.n_rows, other.n_cols)
        other = other.transposed()
        for i in range(self.n_rows):
            for j in range(other.n_rows):  # transposed
                result[i][j] = dot(self[i], other[j])
        return result
