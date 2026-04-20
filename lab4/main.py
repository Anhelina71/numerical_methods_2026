import math

def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def dM_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

def derivative(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

def main():
    t = 1.0

    exact = dM_exact(t)
    print("Точне значення похідної в точці t = 1:", round(exact, 6))

    print("\nЗалежність похибки від кроку h:")

    h_values = [0.1, 0.01, 0.001]
    #best_h = None
    min_error = float("inf")

    for h in h_values:
        approx = derivative(t, h)
        error = abs(approx - exact)

        print(f"h = {h}  D(h) = {approx:.6f}  error = {error:.6f}")

        if error < min_error:
            min_error = error
            best_h = h
    print(f"\n Оптимальний крок h {best_h}")

    h = 0.01
    print("\nКрок h =", h)

    D_h = derivative(t, h)
    D_h2 = derivative(t, h / 2)

    print("\nПохідна для двох кроків:")
    print("D(h)   =", round(D_h, 6))
    print("D(h/2) =", round(D_h2, 6))

    error_h = abs(D_h - exact)

    print("\nПохибка в кроці  h:")
    print("Похибка =", round(error_h, 6))

    p = 2

    D_rr = D_h2 + (D_h2 - D_h) / (2**p - 1)
    error_rr = abs(D_rr - exact)

    print("\nМетод Рунге-Ромберга:")
    print("Уточнене значення похідної =", round(D_rr, 6))
    print("Похибка = ", round(error_rr, 6))

    D_h4 = derivative(t, h / 4)

    print("\nМетод Ейткена:")
    print("D(h)   =", round(D_h, 6))
    print("D(h/2) =", round(D_h2, 6))
    print("D(h/4) =", round(D_h4, 6))

    D_aitken = D_h - (D_h2 - D_h)**2 / (D_h4 - 2 * D_h2 + D_h)
    error_aitken = abs(D_aitken - exact)

    p_est = math.log(abs((D_h2 - D_h) / (D_h4 - D_h2))) / math.log(2)

    print("\nУточнене значення похідної = ", round(D_aitken, 6))
    print("Похибка =", round(error_aitken, 6))
    print("Оцінка порядку точності p = ", round(p_est, 2))

main()