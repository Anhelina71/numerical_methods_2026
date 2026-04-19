"""
Допоміжні функції:
  - зчитування матриці A та вектора B з файлів
  - множення матриці на вектор
  - норми вектора та матриці
"""

import numpy as np
import math


# ─────────────────────────────────────────────
# Введення / виведення
# ─────────────────────────────────────────────

def read_matrix(filename: str) -> np.ndarray:
    """
    Зчитує матрицю n×n з текстового файлу.
    Перший рядок — розмірність n, далі n рядків по n чисел.
    """
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        A = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            A.append(row)
    return np.array(A, dtype=float)


def read_vector(filename: str) -> np.ndarray:
    """
    Зчитує вектор розмірності n з текстового файлу.
    Перший рядок — n, далі n чисел (по одному в рядку).
    """
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        b = [float(f.readline()) for _ in range(n)]
    return np.array(b, dtype=float)


# ─────────────────────────────────────────────
# Лінійна алгебра (власні реалізації)
# ─────────────────────────────────────────────

def mat_vec_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Обчислює добуток матриці A на вектор x: y = A·x."""
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        y[i] = s
    return y


def vec_norm_inf(x: np.ndarray) -> float:
    """Норма вектора: max|x_i|  (рівномірна / Чебишова)."""
    return float(np.max(np.abs(x)))


def vec_norm_2(x: np.ndarray) -> float:
    """Евклідова норма вектора: sqrt(sum x_i^2)."""
    return float(math.sqrt(sum(v * v for v in x)))


def mat_norm_row(A: np.ndarray) -> float:
    """
    Рядкова норма матриці (нескінченна норма):
    ||A||_inf = max_i  sum_j |a_ij|
    """
    n = A.shape[0]
    return float(max(sum(abs(A[i, j]) for j in range(n)) for i in range(n)))


def mat_norm_col(A: np.ndarray) -> float:
    """
    Стовпцева норма матриці (1-норма):
    ||A||_1 = max_j  sum_i |a_ij|
    """
    n = A.shape[0]
    return float(max(sum(abs(A[i, j]) for i in range(n)) for j in range(n)))


def mat_norm_frob(A: np.ndarray) -> float:
    """Норма Фробеніуса: sqrt(sum_{i,j} a_ij^2)."""
    return float(math.sqrt(sum(A[i, j] ** 2
                               for i in range(A.shape[0])
                               for j in range(A.shape[1]))))
