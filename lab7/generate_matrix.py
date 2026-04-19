
import numpy as np
import random

def generate_diag_dominant_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.uniform(-10.0, 10.0, size=(n, n))

    # Забезпечуємо діагональне переважання
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        # Діагональний елемент = сума модулів позадіагональних + запас
        A[i, i] = row_sum + rng.uniform(1.0, 5.0)

    return A


def compute_b_vector(A: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    return A @ x_true


def save_matrix(A: np.ndarray, filename: str) -> None:
    n = A.shape[0]
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{val:.10f}" for val in row) + "\n")
    print(f"Матриця збережена у файл: {filename}")


def save_vector(b: np.ndarray, filename: str) -> None:
    n = len(b)
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for val in b:
            f.write(f"{val:.10f}\n")
    print(f"Вектор збережений у файл: {filename}")


def main():
    n = 100
    x_true_val = 2.5
    x_true = np.full(n, x_true_val)

    print(f"Генерація матриці {n}×{n} з діагональним переважанням...")
    A = generate_diag_dominant_matrix(n)

    print(f"Обчислення вектора B (x_true = {x_true_val} для всіх i)...")
    b = compute_b_vector(A, x_true)

    save_matrix(A, "matrix_A.txt")
    save_vector(b, "vector_B.txt")

    # Перевірка діагонального переважання
    diag_dom = all(
        abs(A[i, i]) > sum(abs(A[i, j]) for j in range(n) if j != i)
        for i in range(n)
    )
    print(f"Діагональне переважання виконується: {diag_dom}")
    print("Генерацію завершено.\n")


if __name__ == "__main__":
    main()