"""
Головна програма — пункти 3-4 ходу роботи + 4 графіки порівняння методів.
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_FILE = os.path.join(BASE_DIR, "matrix_A.txt")
VECTOR_FILE  = os.path.join(BASE_DIR, "vector_B.txt")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import read_matrix, read_vector, vec_norm_inf, mat_norm_row
from iterative_solvers import simple_iteration, jacobi, seidel

COLORS  = {"Проста ітерація": "#378ADD", "Якобі": "#7F77DD", "Зейдель": "#1D9E75"}
MARKERS = {"Проста ітерація": "o",       "Якобі": "s",       "Зейдель": "^"}


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_method(name, solver_fn, A, b, x0, eps, x_true):
    print(f"\n--- {name} ---")
    t0 = time.perf_counter()
    x, iters, converged, h_diff, h_res = solver_fn(A, b, x0, eps=eps)
    elapsed = time.perf_counter() - t0
    residual = vec_norm_inf(A @ x - b)
    error    = vec_norm_inf(x - x_true)
    print(f"  Статус        : {'збіжний' if converged else 'НЕ збіжний'}")
    print(f"  Ітерацій      : {iters}")
    print(f"  Час (сек)     : {elapsed:.4f}")
    print(f"  ||Ax-b||_inf  : {residual:.4e}")
    print(f"  ||x-x_true||  : {error:.4e}")
    return x, iters, elapsed, residual, error, h_diff, h_res


def build_plots(results):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Порівняння ітераційних методів розв'язку СЛАР  (n=100, eps=1e-14)",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.33)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    names = list(results.keys())

    # ── 1. Збіжність ||x^{k+1}-x^k|| ────────────────────────────
    ax1.set_title("Збіжність: ||x^{k+1} − x^k||", fontsize=11)
    for name in names:
        h = results[name][5]   # h_diff
        ax1.semilogy(range(1, len(h)+1), h,
                     color=COLORS[name], marker=MARKERS[name],
                     markevery=max(1, len(h)//8),
                     markersize=5, linewidth=1.8, label=name)
    ax1.axhline(1e-14, color="gray", linestyle="--", linewidth=0.9, label="eps=1e-14")
    ax1.set_xlabel("Ітерація k"); ax1.set_ylabel("||x^{k+1} − x^k||", fontsize=9)
    ax1.legend(fontsize=9); ax1.grid(True, which="both", alpha=0.3)

    # ── 2. Нев'язка ||Ax^k - b|| ─────────────────────────────────
    ax2.set_title("Нев'язка: ||Ax^k − b||", fontsize=11)
    for name in names:
        h = results[name][6]   # h_res
        ax2.semilogy(range(1, len(h)+1), h,
                     color=COLORS[name], marker=MARKERS[name],
                     markevery=max(1, len(h)//8),
                     markersize=5, linewidth=1.8, label=name)
    ax2.axhline(1e-14, color="gray", linestyle="--", linewidth=0.9, label="eps=1e-14")
    ax2.set_xlabel("Ітерація k"); ax2.set_ylabel("||Ax^k − b||", fontsize=9)
    ax2.legend(fontsize=9); ax2.grid(True, which="both", alpha=0.3)

    # ── 3. Кількість ітерацій ─────────────────────────────────────
    ax3.set_title("Кількість ітерацій до збіжності", fontsize=11)
    iters_vals = [results[n][1] for n in names]
    bars = ax3.bar(names, iters_vals,
                   color=[COLORS[n] for n in names],
                   edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, iters_vals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    ax3.set_ylabel("Кількість ітерацій")
    ax3.set_ylim(0, max(iters_vals) * 1.22)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Час виконання ──────────────────────────────────────────
    ax4.set_title("Час виконання", fontsize=11)
    times = [results[n][2] for n in names]
    bars2 = ax4.bar(names, times,
                    color=[COLORS[n] for n in names],
                    edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars2, times):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(times)*0.01,
                 f"{val:.3f} с", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax4.set_ylabel("Час (с)")
    ax4.set_ylim(0, max(times) * 1.25)
    ax4.grid(axis="y", alpha=0.3)

    out_path = os.path.join(BASE_DIR, "comparison_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафіки збережено: {out_path}")
    plt.show()


def main():
    # Генерація якщо файлів нема
    if not os.path.exists(MATRIX_FILE) or not os.path.exists(VECTOR_FILE):
        print("Файли не знайдено. Запускаємо generate_matrix.py...\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_matrix", os.path.join(BASE_DIR, "generate_matrix.py"))
        gm = importlib.util.module_from_spec(spec)
        old = os.getcwd(); os.chdir(BASE_DIR)
        spec.loader.exec_module(gm); gm.main()
        os.chdir(old); print()

    print_section("Зчитування матриці A та вектора B")
    A = read_matrix(MATRIX_FILE)
    b = read_vector(VECTOR_FILE)
    n = len(b)
    print(f"  Файл матриці : {MATRIX_FILE}")
    print(f"  Файл вектора : {VECTOR_FILE}")
    print(f"  Розмірність  : {n}x{n}")
    print(f"  ||A||_inf    : {mat_norm_row(A):.6f}")

    x_true = np.full(n, 2.5)

    print_section("Початкове наближення  x0[i] = 1/(1+i)")
    x0 = np.array([1.0 / (1 + i) for i in range(n)])
    print(f"  x0[0]  = {x0[0]:.6f}")
    print(f"  x0[50] = {x0[50]:.6f}")
    print(f"  x0[99] = {x0[99]:.6f}")
    print(f"  ||x0 - x_true||_inf = {vec_norm_inf(x0 - x_true):.6f}")

    eps = 1e-14
    print_section(f"Розв'язок СЛАР трьома методами, eps = {eps:.0e}")

    results = {}
    for name, fn in [("Проста ітерація", simple_iteration),
                     ("Якобі",           jacobi),
                     ("Зейдель",         seidel)]:
        out = run_method(name, fn, A, b, x0.copy(), eps, x_true)
        results[name] = out   # (x, iters, elapsed, residual, error, h_diff, h_res)

    print_section("Порівняльна таблиця результатів")
    hdr = f"{'Метод':<22} {'Ітерацій':>10} {'||Ax-b||':>14} {'||x-x*||':>14} {'Час(с)':>10}"
    print(hdr); print("-" * len(hdr))
    for name, (_, iters, elapsed, res, err, _, _) in results.items():
        print(f"{name:<22} {iters:>10} {res:>14.4e} {err:>14.4e} {elapsed:>10.4f}")

    print_section("Побудова графіків...")
    build_plots(results)


if __name__ == "__main__":
    main()