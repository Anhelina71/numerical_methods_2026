import numpy as np

# Константи за завданням
N = 100
X_TRUE_VAL = 2.5
EPS_TARGET = 1e-14


def save_to_file(filename, data):
# Допоміжна функція для запису матриць/векторів у файл (Пункт 1, 2)
    np.savetxt(filename, data, fmt='%0.8f')


def load_from_file(filename):
# Допоміжна функція для зчитування даних з файлу (Пункт 2)
    return np.loadtxt(filename)


# ПУНКТ 1: Генерація вихідних даних
def generate_data():
    # Програма випадковим чином генерує елементи матриці A розмірності n x n (n=100)
    A = np.random.uniform(1, 100, size=(N, N))

    # Задаємо розв'язок системи рівнянь (наприклад, що всі x_i = 2,5)
    x_true = np.full(N, X_TRUE_VAL)

    # Обчислюємо вектор вільних членів за формулою b_i = sum(a_ij * x_j)
    b = A @ x_true

    # Записуємо отриману матрицю A та вектор B у текстові файли
    save_to_file('matrix_A.txt', A)
    save_to_file('vector_B.txt', b)
    print("Матриця A та вектор B згенеровані та збережені у файли.")


# ПУНКТ 2: Функції для LU-розкладу та розв'язання
def lu_decomposition(A):
# Знаходження LU-розкладу матриці A (Пункт 2)
    n = len(A)
    L = np.eye(n)  # Нижня трикутна матриця (одиниці на діагоналі)
    U = np.zeros((n, n))  # Верхня трикутна матриця

    for i in range(n):
        for j in range(i, n):
            # Обчислення елементів верхньої трикутної матриці U
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            # Обчислення елементів нижньої трикутної матриці L
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U


def solve_lu(L, U, b):
# Розв'язок системи AX=B за допомогою LU-розкладу (Пункт 2)
    n = len(b)
    # 1. Розв'язуємо Ly = B (пряма підстановка)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    # 2. Розв'язуємо Ux = y (зворотна підстановка)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


def calculate_norm(vector):
# Обчислення норми вектора (Пункт 2)
    # Використовуємо нескінченну норму (максимальний модуль елемента)
    return np.max(np.abs(vector))


# ПУНКТ 5: Ітераційне уточнення розв'язку
def iterative_refinement(A, b, x_initial, L, U, eps_0):
# Пошук уточненого розв'язку СЛАР до заданої точності (Пункт 5)
    x = x_initial.copy()
    iterations = 0

    while True:
        # Обчислення вектора нев'язки r = b - Ax
        r = b - (A @ x)
        error = calculate_norm(r)

        # Перевірка умови досягнення заданої точності (eps_0 = 10^-14)
        if error < eps_0 or iterations > 50:
            break

        # Знаходимо поправку d, розв'язуючи систему Ad = r через LU
        d = solve_lu(L, U, r)

        # Уточнюємо поточний розв'язок: x = x + d
        x = x + d
        iterations += 1

    return x, iterations, error

# ВИКОНАННЯ РОБОТИ

# 1. Генерація даних (Пункт 1)
generate_data()

# 2. Зчитування даних та виконання LU-розкладу (Пункт 2)
A_mat = load_from_file('matrix_A.txt')
B_vec = load_from_file('vector_B.txt')

L_mat, U_mat = lu_decomposition(A_mat)

# Запис LU-розкладу у файл (Пункт 2)
save_to_file('L_matrix.txt', L_mat)
save_to_file('U_matrix.txt', U_mat)

# 3. Розв'язання заданої системи AX = B (Пункт 3)
x_found = solve_lu(L_mat, U_mat, B_vec)

# 4. Оцінка точності знайденого розв'язку (Пункт 4)
# eps = max |sum(a_ij * x_j) - b_i|
residual_vec = (A_mat @ x_found) - B_vec
eps_initial = calculate_norm(residual_vec)

# 5. Уточнення розв'язку до точності 10^-14 (Пункт 5)
x_final, iters_count, final_eps = iterative_refinement(
    A_mat, B_vec, x_found, L_mat, U_mat, EPS_TARGET
)

# 6. Вивід результатів(Пункт 6)
print(f"\nРЕЗУЛЬТАТИ")
print(f"Кількість ітерацій уточнення : {iters_count}")
print(f"Фінальна точність : {final_eps:.2e}")
print(f"Середнє значення X: {np.mean(x_final):.6f} (Очікуване: {X_TRUE_VAL})")