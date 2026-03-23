import csv
import numpy as np
import matplotlib.pyplot as plt

def create_csv():
    data = [
        ["Month", "Temp"],
        [1, -2],
        [2, 0],
        [3, 5],
        [4, 10],
        [5, 15],
        [6, 20],
        [7, 23],
        [8, 22],
        [9, 17],
        [10, 10],
        [11, 5],
        [12, 0],
        [13, -10],
        [14, 3],
        [15, 7],
        [16, 13],
        [17, 19],
        [18, 20],
        [19, 22],
        [20, 21],
        [21, 18],
        [22, 15],
        [23, 10],
        [24, 3]
    ]
    with open("temperature_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print("CSV файл створено")

def read_csv(filename):
    x = []
    y = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return np.array(x), np.array(y)

def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))

    return A
def form_vector(x, y, m):
    b = np.zeros(m + 1)

    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))

    return b

def gauss_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    for k in range(n):
        max_row = np.argmax(abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]

    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)

    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)

    return y_poly

def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)

create_csv()
x, y = read_csv("temperature_data.csv")

max_degree = 4
variances = []

for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)

optimal_m = np.argmin(variances) + 1
A = form_matrix(x, optimal_m)
b_vec = form_vector(x, y, optimal_m)

coef = gauss_solve(A, b_vec)

y_approx = polynomial(x, coef)

x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef)

error = y - y_approx
print("\nТаблиця похибок:")
print("Місяць | Фактична | Апроксимація | Похибка")

for xi, yi, ya, e in zip(x, y, y_approx, error):
    print(f"{xi:5.0f} | {yi:8.2f} | {ya:12.2f} | {e:8.2f}")

print("Дисперсії для різних степенів:")
for i, v in enumerate(variances, start=1):
    print(f"Степінь {i}: {v:.4f}")
degrees = np.arange(1, max_degree + 1)

plt.figure()
plt.plot(degrees, variances, marker='o')
plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")
plt.title("Залежність дисперсії від степеня полінома")
plt.grid()
plt.show()
print("\nОптимальний степінь полінома:", optimal_m)
print("\nПрогноз температур:")
for xi, yi in zip(x_future, y_future):
    print(f"Місяць {xi}: {yi:.2f}")
plt.figure()
plt.plot(x, y, 'o', label="Фактичні дані")
plt.plot(x, y_approx, label="Апроксимація")
plt.plot(x, error, label="Похибка")
plt.xlabel("Місяці")
plt.ylabel("Температура")
plt.legend()
plt.grid()
plt.show()