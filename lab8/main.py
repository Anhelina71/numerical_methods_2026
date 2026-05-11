"""
Лабораторна робота №9
Чисельні методи розв'язування нелінійних рівнянь з одним невідомим
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# Задана трансцендентна функція F(x)
# ============================================================
def F(x):
    return math.sin(x) - 0.5 * x

def dF(x):
    return math.cos(x) - 0.5

def d2F(x):
    return -math.sin(x)

# ============================================================
# 1. Табуляція функції та знаходження наближених коренів
# ============================================================
def tabulate(a, b, h=0.1):
    xs = []
    ys = []
    x = a
    while x <= b + 1e-12:
        xs.append(round(x, 10))
        ys.append(F(x))
        x += h
    return xs, ys

def find_root_intervals(xs, ys):
    """Знайти відрізки зміни знаку (наближені корені)"""
    intervals = []
    for i in range(len(ys) - 1):
        if ys[i] * ys[i+1] < 0:
            intervals.append((xs[i], xs[i+1]))
    return intervals

def save_tabulation(xs, ys, filename="tabulation.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'x':>12}  {'F(x)':>20}\n")
        f.write("-" * 35 + "\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:12.4f}  {y:20.10f}\n")
    print(f"Табуляцію збережено у {filename}")

# ============================================================
# 2–3. Методи уточнення кореня
# ============================================================
EPS = 1e-10

def check_convergence(xn, xprev, fn):
    return abs(fn) < EPS and abs(xn - xprev) < EPS

# --- Метод простої ітерації (релаксація) ---
def simple_iteration(x0, lam=None, max_iter=10000):
    """x_{n+1} = x_n - lambda * F(x_n)"""
    if lam is None:
        # вибираємо lambda так, щоб |1 - lam*F'(x0)| < 1
        lam = 1.0 / abs(dF(x0)) if abs(dF(x0)) > 1e-12 else 0.5
        if dF(x0) > 0:
            lam = abs(lam)
        else:
            lam = -abs(lam)
    x = x0
    for i in range(1, max_iter + 1):
        xnew = x - lam * F(x)
        if check_convergence(xnew, x, F(xnew)):
            return xnew, i
        x = xnew
    return x, max_iter

# --- Метод Ньютона ---
def newton(x0, max_iter=10000):
    x = x0
    for i in range(1, max_iter + 1):
        fx = F(x)
        dfx = dF(x)
        if abs(dfx) < 1e-15:
            break
        xnew = x - fx / dfx
        if check_convergence(xnew, x, F(xnew)):
            return xnew, i
        x = xnew
    return x, max_iter

# --- Метод Чебишева ---
def chebyshev(x0, max_iter=10000):
    x = x0
    for i in range(1, max_iter + 1):
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        if abs(dfx) < 1e-15:
            break
        xnew = x - fx / dfx - (fx**2 * d2fx) / (2 * dfx**3)
        if check_convergence(xnew, x, F(xnew)):
            return xnew, i
        x = xnew
    return x, max_iter

# --- Метод хорд (секущих) ---
def chord(x0, x1, max_iter=10000):
    for i in range(1, max_iter + 1):
        f0, f1 = F(x0), F(x1)
        if abs(f1 - f0) < 1e-15:
            break
        xnew = x1 - f1 * (x1 - x0) / (f1 - f0)
        if check_convergence(xnew, x1, F(xnew)):
            return xnew, i
        x0, x1 = x1, xnew
    return x1, max_iter

# --- Метод парабол (Мюллера) ---
def parabola(x0, x1, x2, max_iter=10000):
    for i in range(1, max_iter + 1):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        # розділені різниці
        d01 = (f1 - f0) / (x1 - x0) if abs(x1 - x0) > 1e-15 else 0
        d12 = (f2 - f1) / (x2 - x1) if abs(x2 - x1) > 1e-15 else 0
        d012 = (d12 - d01) / (x2 - x0) if abs(x2 - x0) > 1e-15 else 0
        # коефіцієнти квадратного рівняння відносно (x - x2)
        A = d012
        B = d12 + d012 * (x2 - x1)
        C = f2
        disc = B**2 - 4 * A * C
        if abs(A) < 1e-15:
            if abs(B) < 1e-15:
                break
            dx = -C / B
        else:
            sq = math.sqrt(abs(disc))
            denom1 = B + sq
            denom2 = B - sq
            dx = -2 * C / denom1 if abs(denom1) >= abs(denom2) else -2 * C / denom2
        xnew = x2 + dx
        if check_convergence(xnew, x2, F(xnew)):
            return xnew, i
        x0, x1, x2 = x1, x2, xnew
    return x2, max_iter

# --- Метод зворотної інтерполяції (три вузли) ---
def inverse_interpolation(x0, x1, x2, max_iter=10000):
    for i in range(1, max_iter + 1):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        # Формула Лагранжа для x(0)
        try:
            L = (x0 * y1 * y2 / ((y0 - y1) * (y0 - y2)) +
                 x1 * y0 * y2 / ((y1 - y0) * (y1 - y2)) +
                 x2 * y0 * y1 / ((y2 - y0) * (y2 - y1)))
        except ZeroDivisionError:
            break
        xnew = L
        if check_convergence(xnew, x2, F(xnew)):
            return xnew, i
        x0, x1, x2 = x1, x2, xnew
    return x2, max_iter

# ============================================================
# Побудова графіка трансцендентної функції
# ============================================================
def plot_transcendental(xs, ys, roots, filename="plot_transcendental.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, 'b-', linewidth=2, label=r'$F(x) = \sin(x) - 0.5x$')
    plt.axhline(0, color='k', linewidth=0.8)
    for r in roots:
        plt.axvline(r, color='r', linestyle='--', alpha=0.5)
        plt.plot(r, 0, 'ro', markersize=8)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Графік функції F(x) = sin(x) − 0.5x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Графік збережено у {filename}")

# ============================================================
# 5–9. Алгебраїчне рівняння 3-го порядку
# Виберемо: x^3 - x^2 + x - 5 = 0
# Один дійсний корінь ≈ 1.516, два комплексних
# ============================================================
POLY_COEFFS = [1, -1, 1, -5]  # a3 x^3 + a2 x^2 + a1 x + a0

def poly_eval(coeffs, x):
    """Обчислення значення многочлена (схема Горнера)"""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

def save_coefficients(coeffs, filename="poly_coeffs.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, coeffs)) + "\n")
    print(f"Коефіцієнти збережено у {filename}")

def load_coefficients(filename="poly_coeffs.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return list(map(float, f.read().split()))

def plot_polynomial(coeffs, filename="plot_polynomial.png"):
    xs = np.linspace(-1, 3, 400)
    ys = [poly_eval(coeffs, x) for x in xs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, 'b-', linewidth=2,
             label=r'$P(x) = x^3 - x^2 + x - 5$')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title('Графік алгебраїчного рівняння третього порядку')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Графік збережено у {filename}")

# ============================================================
# 8. Метод Ньютона зі схемою Горнера для дійсних коренів
# ============================================================
def newton_horner(coeffs, x0, max_iter=10000):
    """
    coeffs = [am, am-1, ..., a1, a0]  (від старшого до вільного)
    Повертає (корінь, кількість ітерацій)
    """
    x = x0
    m = len(coeffs) - 1
    for i in range(1, max_iter + 1):
        # Перша схема Горнера — b0 = F(x)
        b = [0.0] * (m + 1)
        b[m] = coeffs[0]
        for k in range(m - 1, -1, -1):
            b[k] = coeffs[m - k] + x * b[k + 1]
        b0 = b[0]
        # Друга схема Горнера — c1 = F'(x)
        c = [0.0] * m
        c[m - 1] = b[m]
        for k in range(m - 2, -1, -1):
            c[k] = b[k + 1] + x * c[k + 1]
        c1 = c[0]
        if abs(c1) < 1e-15:
            break
        xnew = x - b0 / c1
        if abs(F_poly(coeffs, xnew)) < EPS and abs(xnew - x) < EPS:
            return xnew, i
        x = xnew
    return x, max_iter

def F_poly(coeffs, x):
    return poly_eval(coeffs, x)

# ============================================================
# 9. Метод Ліна для комплексних коренів
# ============================================================
def lin_method(coeffs, alpha0=0.5, beta0=0.5, max_iter=10000):
    """
    Знаходить комплексно-спряжені корені alpha ± i*beta
    """
    m = len(coeffs) - 1  # ступінь
    alpha, beta = alpha0, beta0

    for it in range(1, max_iter + 1):
        p = -2 * alpha
        q = alpha**2 + beta**2

        # Ділення многочлена на (x^2 + p*x + q)
        b = [0.0] * (m + 1)
        b[m] = coeffs[0]
        b[m - 1] = coeffs[1] - p * b[m]
        for k in range(m - 2, 1, -1):
            b[k] = coeffs[m - k] - p * b[k + 1] - q * b[k + 2]
        # залишок: b[1]*x + b[0]
        b[1] = coeffs[m - 1] - p * b[2] - q * b[3]
        b[0] = coeffs[m] - q * b[2]

        r1 = b[1]  # коефіцієнт при x у залишку
        r0 = b[0]  # вільний член залишку

        # Оновлення p і q
        # З умов r1 = 0, r0 = 0 → система для delta_p, delta_q
        # Чисельна Якобіан-апроксимація:
        dp = 1e-7
        dq = 1e-7

        def residuals(pp, qq):
            bb = [0.0] * (m + 1)
            bb[m] = coeffs[0]
            bb[m - 1] = coeffs[1] - pp * bb[m]
            for k in range(m - 2, 1, -1):
                bb[k] = coeffs[m - k] - pp * bb[k + 1] - qq * bb[k + 2]
            bb[1] = coeffs[m - 1] - pp * bb[2] - qq * bb[3]
            bb[0] = coeffs[m] - qq * bb[2]
            return bb[1], bb[0]

        r1p, r0p = residuals(p + dp, q)
        r1q, r0q = residuals(p, q + dq)
        J = ((r1p - r1) / dp * (r0q - r0) / dq -
             (r1q - r1) / dq * (r0p - r0) / dp)
        if abs(J) < 1e-30:
            break
        dp_new = (-(r1) * (r0q - r0) / dq + (r0) * (r1q - r1) / dq) / J
        dq_new = (-(r0) * (r1p - r1) / dp + (r1) * (r0p - r0) / dp) / J

        p_new = p + dp_new
        q_new = q + dq_new

        alpha_new = -p_new / 2
        disc = q_new - alpha_new**2
        beta_new = math.sqrt(abs(disc)) if disc >= 0 else 0.0

        if abs(alpha_new - alpha) < EPS and abs(beta_new - beta) < EPS:
            return alpha_new, beta_new, it
        alpha, beta = alpha_new, beta_new

    return alpha, beta, max_iter

# ============================================================
# ГОЛОВНА ПРОГРАМА
# ============================================================
def main():
    print("=" * 60)
    print("ЛАБОРАТОРНА РОБОТА №9")
    print("Чисельні методи розв'язування нелінійних рівнянь")
    print("=" * 60)

    # 1. Табуляція
    a, b = -8.0, 8.0
    xs, ys = tabulate(a, b, h=0.1)
    save_tabulation(xs, ys)

    intervals = find_root_intervals(xs, ys)
    print(f"\nЗнайдено {len(intervals)} відрізків зміни знаку:")
    for iv in intervals:
        print(f"  [{iv[0]:.2f}, {iv[1]:.2f}]")

    # Виберемо два корені з різною поведінкою:
    # Нульовий тривіальний корінь — пропускаємо (F(0) = 0 точно)
    # Корінь ≈ 1.9 (зростання) і корінь ≈ -1.9 (спадання)
    x0_rise  = 1.5   # початкове наближення (зростаюча гілка)
    x0_fall  = -1.5  # початкове наближення (спадаюча гілка)

    print(f"\nПочаткові наближення: x₀¹ = {x0_rise}, x₀² = {x0_fall}")

    # Побудова графіка
    plot_transcendental(xs, ys, [], filename="plot_transcendental.png")

    # ---- Метод простої ітерації ----
    print("\n--- МЕТОД ПРОСТОЇ ІТЕРАЦІЇ ---")
    for label, x0 in [("Корінь (зростання)", x0_rise),
                      ("Корінь (спадання)",  x0_fall)]:
        root, nit = simple_iteration(x0)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ---- Метод Ньютона ----
    print("\n--- МЕТОД НЬЮТОНА ---")
    for label, x0 in [("Корінь (зростання)", x0_rise),
                      ("Корінь (спадання)",  x0_fall)]:
        root, nit = newton(x0)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ---- Метод Чебишева ----
    print("\n--- МЕТОД ЧЕБИШЕВА ---")
    for label, x0 in [("Корінь (зростання)", x0_rise),
                      ("Корінь (спадання)",  x0_fall)]:
        root, nit = chebyshev(x0)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ---- Метод хорд ----
    print("\n--- МЕТОД ХОРД ---")
    for label, x0, x1 in [("Корінь (зростання)", 1.0, 2.0),
                           ("Корінь (спадання)",  -2.0, -1.0)]:
        root, nit = chord(x0, x1)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ---- Метод парабол ----
    print("\n--- МЕТОД ПАРАБОЛ ---")
    for label, x0, x1, x2 in [("Корінь (зростання)", 1.0, 1.5, 2.0),
                               ("Корінь (спадання)",  -2.0, -1.5, -1.0)]:
        root, nit = parabola(x0, x1, x2)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ---- Метод зворотної інтерполяції ----
    print("\n--- МЕТОД ЗВОРОТНОЇ ІНТЕРПОЛЯЦІЇ ---")
    for label, x0, x1, x2 in [("Корінь (зростання)", 1.0, 1.5, 2.0),
                               ("Корінь (спадання)",  -2.0, -1.5, -1.0)]:
        root, nit = inverse_interpolation(x0, x1, x2)
        print(f"  {label}: x* = {root:.10f}, F(x*) = {F(root):.2e}, ітерацій: {nit}")

    # ============================================================
    # Алгебраїчне рівняння
    # ============================================================
    print("\n" + "=" * 60)
    print("АЛГЕБРАЇЧНЕ РІВНЯННЯ: x³ - x² + x - 5 = 0")
    print("=" * 60)

    save_coefficients(POLY_COEFFS)
    coeffs = load_coefficients()
    print(f"Зчитані коефіцієнти: {coeffs}")

    plot_polynomial(coeffs, "plot_polynomial.png")

    # 8. Метод Ньютона зі схемою Горнера
    print("\n--- МЕТОД НЬЮТОНА (схема Горнера) ---")
    real_root, nit = newton_horner(POLY_COEFFS, x0=2.0)
    print(f"  Дійсний корінь: x* = {real_root:.10f}")
    print(f"  P(x*) = {poly_eval(POLY_COEFFS, real_root):.2e}")
    print(f"  Ітерацій: {nit}")

    # 9. Метод Ліна
    print("\n--- МЕТОД ЛІНА (комплексні корені) ---")
    alpha, beta, nit = lin_method(POLY_COEFFS, alpha0=0.0, beta0=1.0)
    z1 = complex(alpha, beta)
    z2 = complex(alpha, -beta)
    print(f"  Комплексні корені: {z1:.6f}  і  {z2:.6f}")
    # перевірка
    def poly_complex(coeffs, z):
        r = 0+0j
        for c in coeffs:
            r = r * z + c
        return r
    print(f"  P(z1) = {poly_complex(POLY_COEFFS, z1):.2e}")
    print(f"  P(z2) = {poly_complex(POLY_COEFFS, z2):.2e}")
    print(f"  Ітерацій: {nit}")

    # Зведена таблиця ітерацій
    print("\n" + "=" * 60)
    print("ЗВЕДЕНА ТАБЛИЦЯ ІТЕРАЦІЙ (точність 1e-10)")
    print("=" * 60)
    methods = ["Проста ітерація", "Ньютона", "Чебишева",
               "Хорд", "Парабол", "Зворот. інтерп."]
    iters_rise = []
    iters_fall = []

    iters_rise.append(simple_iteration(x0_rise)[1])
    iters_fall.append(simple_iteration(x0_fall)[1])
    iters_rise.append(newton(x0_rise)[1])
    iters_fall.append(newton(x0_fall)[1])
    iters_rise.append(chebyshev(x0_rise)[1])
    iters_fall.append(chebyshev(x0_fall)[1])
    iters_rise.append(chord(1.0, 2.0)[1])
    iters_fall.append(chord(-2.0, -1.0)[1])
    iters_rise.append(parabola(1.0, 1.5, 2.0)[1])
    iters_fall.append(parabola(-2.0, -1.5, -1.0)[1])
    iters_rise.append(inverse_interpolation(1.0, 1.5, 2.0)[1])
    iters_fall.append(inverse_interpolation(-2.0, -1.5, -1.0)[1])

    print(f"{'Метод':<25} {'Корінь (зрост.)',6} {'Корінь (спад.)',6}")
    print("-" * 50)
    for m, ir, ifl in zip(methods, iters_rise, iters_fall):
        print(f"  {m:<23} {ir:>6}          {ifl:>6}")

if __name__ == "__main__":
    main()