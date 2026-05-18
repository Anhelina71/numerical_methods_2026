"""
Лабораторна робота №10
Частина 1: Метод прогнозу і корекції Адамса 2-го порядку
Частина 2: Метод Рунге-Кутта 4-го порядку

Розглядається задача Коші:
    y' = f(x, y),  y(x0) = y0,  x ∈ [a, b]

Приклад рівняння: y' = y - x^2 + 1,  y(0) = 0.5
Точний розв'язок:  y(x) = (x+1)^2 - 0.5*exp(x)
"""

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Визначення задачі
# ──────────────────────────────────────────────

def f(x, y):
    """Права частина ОДР: y' = f(x, y)"""
    return y - x**2 + 1


def exact(x):
    """Точний розв'язок"""
    return (x + 1)**2 - 0.5 * np.exp(x)


# Відрізок та початкова умова
a, b = 0.0, 2.0
y0 = 0.5          # y(0) = 0.5
h_default = 0.1   # крок за замовчуванням
eps = 1e-5        # задана точність


# ══════════════════════════════════════════════
#   ЧАСТИНА 2 — Рунге-Кутта 4-го порядку
# ══════════════════════════════════════════════

def runge_kutta4_step(f, x, y, h):
    """Один крок методу Рунге-Кутта 4-го порядку."""
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def runge_kutta4(f, a, b, y0, h):
    """
    Метод Рунге-Кутта 4-го порядку на рівномірній сітці.
    Повертає масиви вузлів x та значень y.
    """
    xs = [a]
    ys = [y0]
    x = a
    y = y0
    while x + h <= b + 1e-12:
        y = runge_kutta4_step(f, x, y, h)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def local_error_exact_rk4(f, a, b, y0, h):
    """Локальна похибка РК4: |y_num - y_exact|."""
    xs, ys = runge_kutta4(f, a, b, y0, h)
    return xs, np.abs(ys - exact(xs))


def runge_error_rk4(f, x, y, h):
    """
    Оцінка локальної похибки методом Рунге (два кроки h/2 vs один крок h).
    |err| ≈ |y(2*h/2) - y(h)| / (2^4 - 1)
    """
    # Один крок розміром h
    y1 = runge_kutta4_step(f, x, y, h)
    # Два кроки розміром h/2
    y_half = runge_kutta4_step(f, x, y, h/2)
    y2 = runge_kutta4_step(f, x + h/2, y_half, h/2)
    return abs(y2 - y1) / (2**4 - 1)


def auto_step_rk4(f, a, b, y0, eps):
    """
    Автоматичний вибір кроку для РК4 методом Рунге.
    Крок подвоюється або вдвічі зменшується залежно від локальної похибки.
    """
    xs = [a]
    ys = [y0]
    hs = []
    x, y = a, y0
    h = (b - a) / 10   # початковий крок
    C = 0.9             # константа запасу

    while x < b - 1e-12:
        if x + h > b:
            h = b - x
        err = runge_error_rk4(f, x, y, h)

        if err > eps:
            h /= 2          # зменшити крок
            continue

        # Виконати крок
        y = runge_kutta4_step(f, x, y, h)
        x += h
        xs.append(x)
        ys.append(y)
        hs.append(h)

        if err < eps / 32:  # можна збільшити крок
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)


# ══════════════════════════════════════════════
#   ЧАСТИНА 1 — Адамс 2-го порядку (прогноз-корекція)
# ══════════════════════════════════════════════

def adams2(f, a, b, y0, h):
    """
    Метод Адамса прогнозу-корекції 2-го порядку.

    Прогноз (Adams-Bashforth):
        y*_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})

    Корекція (Adams-Moulton):
        y_{n+1} = y_n + h/2 * (f_{n+1}* + f_n)
    """
    # Перший крок — Рунге-Кутта (для розгону)
    xs = [a]
    ys = [y0]
    x0 = a
    x1 = a + h
    y1 = runge_kutta4_step(f, x0, y0, h)
    xs.append(x1)
    ys.append(y1)

    x = x1
    y_prev, y_curr = y0, y1

    while x + h <= b + 1e-12:
        f_prev = f(x - h, y_prev)
        f_curr = f(x,      y_curr)

        # Прогноз
        y_pred = y_curr + h/2 * (3*f_curr - f_prev)

        # Корекція (одна ітерація)
        f_pred = f(x + h, y_pred)
        y_corr = y_curr + h/2 * (f_pred + f_curr)

        x += h
        xs.append(x)
        ys.append(y_corr)

        y_prev = y_curr
        y_curr = y_corr

    return np.array(xs), np.array(ys)


def local_error_exact_adams2(f, a, b, y0, h):
    """Локальна похибка Адамса 2: |y_num - y_exact|."""
    xs, ys = adams2(f, a, b, y0, h)
    return xs, np.abs(ys - exact(xs))


def adams2_runge_error(f, a, b, y0, h):
    """
    Оцінка похибки методом Рунге для Адамса 2-го порядку.
    Порівнюємо розв'язок з кроком h та 2h.
    """
    xs_h,  ys_h  = adams2(f, a, b, y0, h)
    xs_2h, ys_2h = adams2(f, a, b, y0, 2*h)

    # Спільні вузли (кожен другий з дрібної сітки)
    errors = []
    xc = []
    j = 0
    for i in range(len(xs_2h)):
        while j < len(xs_h) and abs(xs_h[j] - xs_2h[i]) > 1e-12:
            j += 1
        if j < len(xs_h):
            errors.append(abs(ys_h[j] - ys_2h[i]) / (2**2 - 1))
            xc.append(xs_2h[i])
    return np.array(xc), np.array(errors)


def auto_step_adams2(f, a, b, y0, eps):
    """
    Автоматичний вибір кроку для методу Адамса 2-го порядку.
    При порушенні точності крок зменшується вдвічі (перезапуск із РК4).
    """
    xs = [a]
    ys = [y0]
    hs = []
    h = (b - a) / 10
    x, y = a, y0

    # Стартові значення (два вузли для Адамса)
    x_prev, y_prev = x, y
    x, y = x + h, runge_kutta4_step(f, x, y, h)
    xs.append(x)
    ys.append(y)

    while x < b - 1e-12:
        if x + h > b:
            h = b - x

        f_prev = f(x_prev, y_prev)
        f_curr = f(x, y)

        # Прогноз
        y_pred = y + h/2 * (3*f_curr - f_prev)
        # Корекція
        y_corr = y + h/2 * (f(x + h, y_pred) + f_curr)

        # Оцінка похибки (різниця прогноз-корекція / (2^2 - 1) * 2^2)
        err = abs(y_corr - y_pred) * 2 / (2**2 - 1)

        if err > eps:
            # Зменшити крок вдвічі, перезапустити з РК4
            h /= 2
            x, y = xs[-1], ys[-1]
            x_prev = x - h
            # Наближаємо y_prev зворотньо (один крок РК4 назад неможливий,
            # тому беремо попереднє значення з масиву, якщо є)
            if len(ys) >= 2:
                y_prev = ys[-2]
                x_prev = xs[-2]
            continue

        x_prev, y_prev = x, y
        x += h
        xs.append(x)
        ys.append(y_corr)
        hs.append(h)
        y = y_corr

        if err < eps / 4:
            h *= 2  # збільшити крок

    return np.array(xs), np.array(ys), np.array(hs)


# ══════════════════════════════════════════════
#   ПОБУДОВА ГРАФІКІВ
# ══════════════════════════════════════════════

def plot_all():
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Лабораторна робота №10 — Чисельне розв'язання ОДР", fontsize=14)

    x_ex = np.linspace(a, b, 500)
    y_ex = exact(x_ex)

    h = h_default

    # ── Розв'язки ──────────────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    xs_rk, ys_rk = runge_kutta4(f, a, b, y0, h)
    xs_ad, ys_ad = adams2(f, a, b, y0, h)
    ax1.plot(x_ex, y_ex, 'k-', label='Точний', linewidth=2)
    ax1.plot(xs_rk, ys_rk, 'b--o', markersize=3, label=f'РК4 (h={h})')
    ax1.plot(xs_ad, ys_ad, 'r--s', markersize=3, label=f'Адамс2 (h={h})')
    ax1.set_title('Розв\'язки')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(fontsize=7)
    ax1.grid(True)

    # ── Похибка РК4 (точна) ──────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    xs_err, err_rk = local_error_exact_rk4(f, a, b, y0, h)
    ax2.semilogy(xs_err, err_rk, 'b-o', markersize=3)
    ax2.set_title('РК4: |y_num − y_exact|')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Похибка')
    ax2.grid(True)

    # ── Похибка РК4 (метод Рунге) ────────────
    ax3 = fig.add_subplot(3, 3, 3)
    runge_errs = [runge_error_rk4(f, xs_rk[i], ys_rk[i], h)
                  for i in range(len(xs_rk) - 1)]
    ax3.semilogy(xs_rk[:-1], runge_errs, 'g-^', markersize=3)
    ax3.set_title('РК4: оцінка похибки (метод Рунге)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Похибка')
    ax3.grid(True)

    # ── Похибка РК4 vs кроку ─────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    h_vals = [0.4, 0.2, 0.1, 0.05, 0.025]
    max_errs_rk = []
    for hv in h_vals:
        _, errs = local_error_exact_rk4(f, a, b, y0, hv)
        max_errs_rk.append(np.max(errs))
    ax4.loglog(h_vals, max_errs_rk, 'b-o')
    # Лінія O(h^4)
    h_ref = np.array(h_vals)
    ax4.loglog(h_ref, max_errs_rk[0] * (h_ref / h_ref[0])**4, 'k--', label='O(h⁴)')
    ax4.set_title('РК4: max|похибка| від кроку h')
    ax4.set_xlabel('h')
    ax4.set_ylabel('max|err|')
    ax4.legend()
    ax4.grid(True, which='both')

    # ── Автоматичний крок РК4 ────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    xs_auto_rk, ys_auto_rk, hs_auto_rk = auto_step_rk4(f, a, b, y0, eps)
    ax5.step(xs_auto_rk[:-1], hs_auto_rk, 'b-', where='post')
    ax5.set_title(f'РК4: автоматичний крок (ε={eps})')
    ax5.set_xlabel('x')
    ax5.set_ylabel('h')
    ax5.grid(True)

    # ── Похибка Адамса (точна) ───────────────
    ax6 = fig.add_subplot(3, 3, 6)
    xs_ad_err, err_ad = local_error_exact_adams2(f, a, b, y0, h)
    ax6.semilogy(xs_ad_err, err_ad + 1e-18, 'r-o', markersize=3)
    ax6.set_title('Адамс2: |y_num − y_exact|')
    ax6.set_xlabel('x')
    ax6.set_ylabel('Похибка')
    ax6.grid(True)

    # ── Похибка Адамса (метод Рунге) ─────────
    ax7 = fig.add_subplot(3, 3, 7)
    xc, err_runge_ad = adams2_runge_error(f, a, b, y0, h)
    ax7.semilogy(xc, err_runge_ad + 1e-18, 'r-^', markersize=3)
    ax7.set_title('Адамс2: оцінка похибки (метод Рунге)')
    ax7.set_xlabel('x')
    ax7.set_ylabel('Похибка')
    ax7.grid(True)

    # ── Похибка Адамса vs кроку ──────────────
    ax8 = fig.add_subplot(3, 3, 8)
    max_errs_ad = []
    for hv in h_vals:
        _, errs = local_error_exact_adams2(f, a, b, y0, hv)
        max_errs_ad.append(np.max(errs))
    ax8.loglog(h_vals, max_errs_ad, 'r-o')
    ax8.loglog(h_ref, max_errs_ad[0] * (h_ref / h_ref[0])**2, 'k--', label='O(h²)')
    ax8.set_title('Адамс2: max|похибка| від кроку h')
    ax8.set_xlabel('h')
    ax8.set_ylabel('max|err|')
    ax8.legend()
    ax8.grid(True, which='both')

    # ── Автоматичний крок Адамса ─────────────
    ax9 = fig.add_subplot(3, 3, 9)
    xs_auto_ad, ys_auto_ad, hs_auto_ad = auto_step_adams2(f, a, b, y0, eps)
    n_ad = min(len(xs_auto_ad) - 1, len(hs_auto_ad))
    ax9.step(xs_auto_ad[:n_ad], hs_auto_ad[:n_ad], 'r-', where='post')
    ax9.set_title(f'Адамс2: автоматичний крок (ε={eps})')
    ax9.set_xlabel('x')
    ax9.set_ylabel('h')
    ax9.grid(True)

    plt.tight_layout()
    plt.savefig('lab10_plots.png', dpi=150)
    plt.show()
    print("Графіки збережено: lab10_plots.png")


# ══════════════════════════════════════════════
#   ВИВЕДЕННЯ ТАБЛИЦЬ У КОНСОЛЬ
# ══════════════════════════════════════════════

def print_table_rk4(h):
    print(f"\n{'─'*72}")
    print(f"  РК4, крок h = {h}")
    print(f"{'─'*72}")
    print(f"{'x':>8}  {'y_num':>12}  {'y_exact':>12}  {'|err|':>12}")
    print(f"{'─'*72}")
    xs, ys = runge_kutta4(f, a, b, y0, h)
    for x, y in zip(xs, ys):
        ye = exact(x)
        print(f"{x:8.4f}  {y:12.8f}  {ye:12.8f}  {abs(y-ye):12.2e}")
    print(f"{'─'*72}")


def print_table_adams2(h):
    print(f"\n{'─'*72}")
    print(f"  Адамс (прогноз-корекція 2-го порядку), крок h = {h}")
    print(f"{'─'*72}")
    print(f"{'x':>8}  {'y_num':>12}  {'y_exact':>12}  {'|err|':>12}")
    print(f"{'─'*72}")
    xs, ys = adams2(f, a, b, y0, h)
    for x, y in zip(xs, ys):
        ye = exact(x)
        print(f"{x:8.4f}  {y:12.8f}  {ye:12.8f}  {abs(y-ye):12.2e}")
    print(f"{'─'*72}")


# ══════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  Лабораторна робота №10")
    print("  y' = y - x^2 + 1,  y(0) = 0.5,  x ∈ [0, 2]")
    print("  Точний розв'язок: y = (x+1)^2 - 0.5*exp(x)")
    print("=" * 72)

    print_table_rk4(h_default)
    print_table_adams2(h_default)

    print("\nАвтоматичний вибір кроку (РК4):")
    xs_a, ys_a, hs_a = auto_step_rk4(f, a, b, y0, eps)
    print(f"  Вузлів: {len(xs_a)},  кроків: min={hs_a.min():.4f}, max={hs_a.max():.4f}")
    print(f"  Макс. похибка: {np.max(np.abs(ys_a - exact(xs_a))):.2e}")

    print("\nАвтоматичний вибір кроку (Адамс2):")
    xs_b, ys_b, hs_b = auto_step_adams2(f, a, b, y0, eps)
    print(f"  Вузлів: {len(xs_b)},  кроків: min={hs_b.min():.4f}, max={hs_b.max():.4f}")
    print(f"  Макс. похибка: {np.max(np.abs(ys_b - exact(xs_b))):.2e}")

    plot_all()