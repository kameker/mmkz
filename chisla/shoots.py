from pathlib import Path

import numpy as np

if __package__:
    from .cauchy import cauchy1
else:
    from cauchy import cauchy1


def equation(x, u):
    y, y_prime = u
    return np.array([y_prime, y_prime + x])


def shooting(h=0.01, eps=0.01, left=-2.0, right=0.0):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть положительным конечным числом")
    if not np.isfinite([left, right]).all() or left >= right:
        raise ValueError("Нужен конечный отрезок left < right")

    def shoot(alpha):
        x, u = cauchy1(equation, [-1.0, alpha], 0.0, 1.0, h)
        return u[-1, 0] + 2.0, x, u

    f_left, x, u = shoot(left)
    if abs(f_left) <= eps:
        return left, x, u

    f_right, x, u = shoot(right)
    if abs(f_right) <= eps:
        return right, x, u
    if f_left * f_right > 0:
        raise ValueError("Невязка должна менять знак на концах отрезка")

    for _ in range(100):
        alpha = (left + right) / 2
        residual, x, u = shoot(alpha)
        if abs(residual) <= eps:
            return alpha, x, u
        if f_left * residual < 0:
            right = alpha
        else:
            left = alpha
            f_left = residual

    raise RuntimeError("Заданная точность не достигнута за 100 итераций")


def main():
    alpha, x, u = shooting(h=0.01, eps=0.01)

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "shoots_solution.csv"
    np.savetxt(
        output_path,
        np.column_stack((x, u)),
        delimiter=",",
        header="x,y,y_prime",
        comments="",
    )

    print(f"y'(0) = {alpha:.8f}")
    print(f"y(0) = {u[0, 0]:.8f}")
    print(f"y(1) = {u[-1, 0]:.8f}")
    print(f"Невязка |y(1) + 2| = {abs(u[-1, 0] + 2):.8f}")
    print(f"Точки сохранены: {output_path}")


if __name__ == "__main__":
    main()
