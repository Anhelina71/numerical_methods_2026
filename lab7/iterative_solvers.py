"""
Ітераційні методи розв'язку СЛАР.
Кожен метод повертає (x, iterations, converged, history_diff, history_res)
  history_diff - ||x^{k+1} - x^k|| на кожній ітерації
  history_res  - ||A x^k - b||     на кожній ітерації
"""

import numpy as np
from utils import vec_norm_inf


def simple_iteration(A, b, x0, eps=1e-14, max_iter=10_000):
    n = len(b)
    A_norm = float(np.max(np.sum(np.abs(A), axis=1)))
    tau = 1.0 / A_norm
    C = np.eye(n) - tau * A
    d = tau * b
    C_norm = float(np.max(np.sum(np.abs(C), axis=1)))
    print(f"[Проста ітерація] tau={tau:.6e}, ||C||_inf={C_norm:.6f}")
    x = x0.copy()
    history_diff, history_res = [], []
    for k in range(1, max_iter + 1):
        x_new = C @ x + d
        diff = vec_norm_inf(x_new - x)
        res  = vec_norm_inf(A @ x_new - b)
        history_diff.append(diff)
        history_res.append(res)
        x = x_new
        if diff < eps:
            return x, k, True, history_diff, history_res
    return x, max_iter, False, history_diff, history_res


def jacobi(A, b, x0, eps=1e-14, max_iter=10_000):
    n = len(b)
    x = x0.copy()
    history_diff, history_res = [], []
    if any(A[i, i] == 0 for i in range(n)):
        raise ValueError("Нульовий діагональний елемент.")
    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]
        diff = vec_norm_inf(x_new - x)
        res  = vec_norm_inf(A @ x_new - b)
        history_diff.append(diff)
        history_res.append(res)
        x = x_new
        if diff < eps:
            return x, k, True, history_diff, history_res
    return x, max_iter, False, history_diff, history_res


def seidel(A, b, x0, eps=1e-14, max_iter=10_000):
    n = len(b)
    x = x0.copy()
    history_diff, history_res = [], []
    if any(A[i, i] == 0 for i in range(n)):
        raise ValueError("Нульовий діагональний елемент.")
    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            s_left  = sum(A[i, j] * x_new[j] for j in range(i))
            s_right = sum(A[i, j] * x[j]     for j in range(i + 1, n))
            x_new[i] = (b[i] - s_left - s_right) / A[i, i]
        diff = vec_norm_inf(x_new - x)
        res  = vec_norm_inf(A @ x_new - b)
        history_diff.append(diff)
        history_res.append(res)
        x = x_new
        if diff < eps:
            return x, k, True, history_diff, history_res
    return x, max_iter, False, history_diff, history_res