"""
Ітераційні методи розв'язку СЛАР:
  1. Метод простої ітерації
  2. Метод Якобі
  3. Метод Зейделя (Гауса–Зейделя)
"""

import numpy as np
from utils import vec_norm_inf


# ─────────────────────────────────────────────
# 1. Метод простої ітерації
# ─────────────────────────────────────────────

def simple_iteration(A: np.ndarray, b: np.ndarray,
                     x0: np.ndarray,
                     eps: float = 1e-14,
                     max_iter: int = 10_000):
    """
    Метод простої ітерації:
        x^{k+1} = x^k - τ·(A·x^k - b),
    де τ вибирається як 2 / (λ_min + λ_max) для симетричної A,
    або безпечно τ = 1 / ||A||_inf.

    Для матриці з діагональним переважанням достатня умова збіжності:
        ||C||_inf < 1,  C = E - τ·A.

    Повертає: (x, iterations, converged)
    """
    n = len(b)

    # Вибір τ: τ = 1 / ||A||_inf  гарантує ||C|| < 1
    A_norm = float(np.max(np.sum(np.abs(A), axis=1)))   # рядкова норма
    tau = 1.0 / A_norm

    # Матриця C = E - τ·A  та вектор d = τ·b
    C = np.eye(n) - tau * A
    d = tau * b

    C_norm = float(np.max(np.sum(np.abs(C), axis=1)))
    print(f"[Проста ітерація] τ={tau:.6e}, ||C||_inf={C_norm:.6f}")

    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = C @ x + d

        diff = vec_norm_inf(x_new - x)
        x = x_new

        if diff < eps:
            return x, k, True

    return x, max_iter, False


# ─────────────────────────────────────────────
# 2. Метод Якобі
# ─────────────────────────────────────────────

def jacobi(A: np.ndarray, b: np.ndarray,
           x0: np.ndarray,
           eps: float = 1e-14,
           max_iter: int = 10_000):
    """
    Метод Якобі:
        x_i^{k+1} = (b_i - sum_{j≠i} a_ij * x_j^k) / a_ii

    Формула в розгорнутому вигляді відповідно до теоретичних відомостей.

    Повертає: (x, iterations, converged)
    """
    n = len(b)
    x = x0.copy()

    # Перевірка наявності нульових діагональних елементів
    if any(A[i, i] == 0 for i in range(n)):
        raise ValueError("Нульовий діагональний елемент — метод Якобі незастосовний.")

    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]

        diff = vec_norm_inf(x_new - x)
        x = x_new

        if diff < eps:
            return x, k, True

    return x, max_iter, False


# ─────────────────────────────────────────────
# 3. Метод Зейделя (Гауса–Зейделя)
# ─────────────────────────────────────────────

def seidel(A: np.ndarray, b: np.ndarray,
           x0: np.ndarray,
           eps: float = 1e-14,
           max_iter: int = 10_000):
    """
    Метод Гауса–Зейделя:
        x_i^{k+1} = (b_i
                      - sum_{j<i} a_ij * x_j^{k+1}   ← вже оновлені
                      - sum_{j>i} a_ij * x_j^k) / a_ii

    Повертає: (x, iterations, converged)
    """
    n = len(b)
    x = x0.copy()

    if any(A[i, i] == 0 for i in range(n)):
        raise ValueError("Нульовий діагональний елемент — метод Зейделя незастосовний.")

    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            # Ліва частина — вже оновлені значення x_new[0..i-1]
            s_left = sum(A[i, j] * x_new[j] for j in range(i))
            # Права частина — ще старі значення x[i+1..n-1]
            s_right = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s_left - s_right) / A[i, i]

        diff = vec_norm_inf(x_new - x)
        x = x_new

        if diff < eps:
            return x, k, True

    return x, max_iter, False