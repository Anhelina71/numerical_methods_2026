"""
Головна програма — пункти 3–4 ходу роботи:

  3. Початкове наближення:  x0[i] = 1.0 / (1 + i),  i = 0..n-1
  4. Розв'язок методами простої ітерації, Якобі, Зейделя з eps = 1e-14.
     Виведення кількості ітерацій та похибки для кожного методу.
"""

import numpy as np
import time
import os
import sys

# ── Шляхи до файлів відносно розташування цього скрипта ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_FILE = os.path.join(BASE_DIR, "matrix_A.txt")
VECTOR_FILE  = os.path.join(BASE_DIR, "vector_B.txt")

# Додаємо папку скрипта до шляху пошуку модулів (для utils та iterative_solvers)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import read_matrix, read_vector, vec_norm_inf, mat_norm_row
from iterative_solvers import simple_iteration, jacobi, seidel


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_method(name, solver_fn, A, b, x0, eps, x_true):
    print(f"\n--- {name} ---")
    t0 = time.perf_counter()
    x, iters, converged = solver_fn(A, b, x0, eps=eps)
    elapsed = time.perf_counter() - t0

    residual = vec_norm_inf(A @ x - b)
    error    = vec_norm_inf(x - x_true)

    status = "збіжний" if converged else "НЕ збіжний (досягнуто max_iter)"
    print(f"  Статус        : {status}")
    print(f"  Ітерацій      : {iters}")
    print(f"  Час (сек)     : {elapsed:.4f}")
    print(f"  ||Ax-b||_inf  : {residual:.4e}")
    print(f"  ||x-x_true||  : {error:.4e}")

    return x, iters


def main():
    # ── Перевірка наявності файлів — якщо нема, генеруємо ────────
    if not os.path.exists(MATRIX_FILE) or not os.path.exists(VECTOR_FILE):
        print("Файли matrix_A.txt / vector_B.txt не знайдено.")
        print(f"Очікується розташування: {BASE_DIR}")
        print("Запускаємо generate_matrix.py автоматично...\n")

        import importlib.util
        gen_path = os.path.join(BASE_DIR, "generate_matrix.py")
        spec = importlib.util.spec_from_file_location("generate_matrix", gen_path)
        gen_module = importlib.util.module_from_spec(spec)
        old_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        spec.loader.exec_module(gen_module)
        gen_module.main()
        os.chdir(old_cwd)
        print()

    # ── Зчитування даних ──────────────────────────────────────────
    print_section("Зчитування матриці A та вектора B")
    A = read_matrix(MATRIX_FILE)
    b = read_vector(VECTOR_FILE)
    n = len(b)
    print(f"  Файл матриці  : {MATRIX_FILE}")
    print(f"  Файл вектора  : {VECTOR_FILE}")
    print(f"  Розмірність системи: {n}x{n}")
    print(f"  ||A||_inf (рядкова норма): {mat_norm_row(A):.6f}")

    x_true = np.full(n, 2.5)

    # ── Пункт 3: початкове наближення ────────────────────────────
    print_section("Початкове наближення  x0[i] = 1/(1+i)")
    x0 = np.array([1.0 / (1 + i) for i in range(n)])
    print(f"  x0[0]  = {x0[0]:.6f}")
    print(f"  x0[50] = {x0[50]:.6f}")
    print(f"  x0[99] = {x0[99]:.6f}")
    print(f"  ||x0 - x_true||_inf = {vec_norm_inf(x0 - x_true):.6f}")

    # ── Пункт 4: розв'язок трьома методами ───────────────────────
    eps = 1e-14
    print_section(f"Розв'язок СЛАР трьома методами, eps = {eps:.0e}")

    run_method("Метод простої ітерації",
               simple_iteration, A, b, x0.copy(), eps, x_true)
    run_method("Метод Якобі",
               jacobi, A, b, x0.copy(), eps, x_true)
    run_method("Метод Зейделя (Гауса-Зейделя)",
               seidel, A, b, x0.copy(), eps, x_true)

    # ── Порівняльна таблиця ───────────────────────────────────────
    print_section("Порівняльна таблиця результатів")
    header = f"{'Метод':<30} {'Ітерацій':>10} {'||Ax-b||':>14} {'||x-x*||':>14} {'Час(с)':>10}"
    print(header)
    print("-" * len(header))

    for name, fn in [("Проста ітерація", simple_iteration),
                     ("Якобі",           jacobi),
                     ("Зейдель",         seidel)]:
        t0 = time.perf_counter()
        x, iters, _ = fn(A, b, x0.copy(), eps=eps)
        elapsed = time.perf_counter() - t0
        res = vec_norm_inf(A @ x - b)
        err = vec_norm_inf(x - x_true)
        print(f"{name:<30} {iters:>10} {res:>14.4e} {err:>14.4e} {elapsed:>10.4f}")

    print()


if __name__ == "__main__":
    main()