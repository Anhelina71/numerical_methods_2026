import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return x**2 + y**2 - 4

def f2(x, y):
    return x - y - 1


# цільова функція
def objective(v):
    x, y = v
    return f1(x, y)**2 + f2(x, y)**2

def rosenbrock(v):
    x, y = v
    return 100 * (y - x**2)**2 + (1 - x)**2

# Метод Хука-Дживса

def exploratory_search(func, base, step):
    x = np.array(base, dtype=float)
    f_base = func(x)

    for i in range(len(x)):
        x_try = x.copy()
        x_try[i] += step[i]

        if func(x_try) < f_base:
            x = x_try
            f_base = func(x)
        else:
            x_try[i] -= 2 * step[i]

            if func(x_try) < f_base:
                x = x_try
                f_base = func(x)

    return x


def hooke_jeeves(func, x0, step0, alpha=2, eps=1e-6, max_iter=500):
    base = np.array(x0, dtype=float)
    step = np.array(step0, dtype=float)

    path = [base.copy()]
    iterations = 0

    while np.max(step) > eps and iterations < max_iter:
        iterations += 1

        new_point = exploratory_search(func, base, step)

        if func(new_point) < func(base):
            while True:
                pattern = new_point + alpha * (new_point - base)
                explored = exploratory_search(func, pattern, step)

                path.append(explored.copy())

                if func(explored) < func(new_point):
                    base = new_point
                    new_point = explored
                else:
                    base = new_point
                    break
        else:
            step = step / 2

        path.append(base.copy())

    return base, func(base), path, iterations

def plot_system():
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)

    X, Y = np.meshgrid(x, y)

    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z1, levels=[0], colors='blue')
    plt.contour(X, Y, Z2, levels=[0], colors='red')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графіки системи нелінійних рівнянь')
    plt.legend(['x²+y²-4=0', 'x-y-1=0'])
    plt.show()

def plot_path(path):
    path = np.array(path)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    Z = (X**2 + Y**2 - 4)**2 + (X - Y - 1)**2

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, 30)
    plt.plot(path[:, 0], path[:, 1], 'ro-')
    plt.grid(True)
    plt.title('Траєкторія спуску')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def save_path(path):
    with open("trajectory.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(path):
            f.write(f"{i+1}: x = {p[0]:.6f}, y = {p[1]:.6f}\n")

def main():
    print()
    print("ТЕСТ НА ФУНКЦІЇ РОЗЕНБРОКА")
    print()

    x0 = [-1.2, 1.0]
    step0 = [0.5, 0.5]

    point, value, path, iters = hooke_jeeves(rosenbrock, x0, step0)

    print("Мінімум:")
    print("x =", point)
    print("f(x) =", value)
    print("Кількість кроків =", iters)

    print()
    print("РОЗВ'ЯЗОК СИСТЕМИ")
    print()

    x0 = [1.0, 1.0]
    step0 = [0.5, 0.5]

    point, value, path, iters = hooke_jeeves(objective, x0, step0)

    print("Розв'язок системи:")
    print("x =", point[0])
    print("y =", point[1])
    print("F(x,y) =", value)
    print("Кількість кроків =", iters)

    save_path(path)
    print("Траєкторія записана у файл trajectory.txt")

    plot_system()
    plot_path(path)

main()