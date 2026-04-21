import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

x = np.linspace(a, b, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(
    x, y,
    label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

def simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    S = y[0] + y[-1]
    S += 4 * np.sum(y[1:-1:2])   # непарні
    S += 2 * np.sum(y[2:-2:2])   # парні

    return S * h / 3

I_exact = simpson(f, a, b, 2000)
print("Еталонне значення інтегралу (Simpson n=2000):", I_exact)

n_values = np.arange(2, 100, 2)
errors = []

for n in n_values:
    I = simpson(f, a, b, n)
    errors.append(abs(I - I_exact))

plt.figure()
plt.plot(n_values, errors)
plt.xlabel("n")
plt.ylabel("Похибка")
plt.title("Залежність похибки від n")
plt.grid()
plt.show()

eps = 1e-5
for n, err in zip(n_values, errors):
    if err < eps:
        print(f"Мінімальне n для точності {eps}: {n}")
        break

n_test = 20
I_test = simpson(f, a, b, n_test)
error_test = abs(I_test - I_exact)

print(f"Похибка при n={n_test}:", error_test)

n = 20
I_h = simpson(f, a, b, n)
I_h2 = simpson(f, a, b, 2 * n)

p = 4

I_rr = I_h2 + (I_h2 - I_h) / (2**p - 1)
error_rr = abs(I_rr - I_exact)

print("Уточнене (Рунге-Ромберг):", I_rr)
print("Похибка Рунге-Ромберга:", error_rr)

n1, n2, n3 = 10, 20, 40

I1 = simpson(f, a, b, n1)
I2 = simpson(f, a, b, n2)
I3 = simpson(f, a, b, n3)

p_est = np.log(abs((I3 - I2) / (I2 - I1))) / np.log(2)

I_aitken = I3 + (I3 - I2) / (2**p_est - 1)
error_aitken = abs(I_aitken - I_exact)

print("Оцінка порядку (Ейткен):", p_est)
print("Уточнене значення (Ейткен):", I_aitken)
print("Похибка Ейткена:", error_aitken)

def adaptive_simpson(f, a, b, eps, max_depth=10):

    def simpson_local(f, a, b):
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4*f(c) + f(b))

    def recurse(f, a, b, eps, whole, depth):
        c = (a + b) / 2
        left = simpson_local(f, a, c)
        right = simpson_local(f, c, b)

        if depth <= 0 or abs(left + right - whole) < 15 * eps:
            return left + right + (left + right - whole) / 15
        return (recurse(f, a, c, eps / 2, left, depth - 1) +
                recurse(f, c, b, eps / 2, right, depth - 1))

    initial = simpson_local(f, a, b)
    return recurse(f, a, b, eps, initial, max_depth)

eps_adapt = 1e-5
I_adapt = adaptive_simpson(f, a, b, eps_adapt)
error_adapt = abs(I_adapt - I_exact)

print("\nАдаптивний результат:", I_adapt)
print("Похибка адаптивного:", error_adapt)