from pathlib import Path

import numpy as np

from cauchy import cauchy1, cauchy2


def third_order_equation(x, u):
    y, y_prime, y_second = u
    return np.array([y_prime, y_second, -y_prime])


def main():
    x0, xn, h = 0.0, 2.0 * np.pi, 0.01
    u0 = np.array([0.0, 1.0, 0.0])

    x_mid, u_mid = cauchy1(third_order_equation, u0, x0, xn, h)
    x_heun, u_heun = cauchy2(third_order_equation, u0, x0, xn, h)

    exact_mid = np.sin(x_mid)
    exact_heun = np.sin(x_heun)
    error_mid = np.max(np.abs(u_mid[:, 0] - exact_mid))
    error_heun = np.max(np.abs(u_heun[:, 0] - exact_heun))

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    np.savetxt(
        output_dir / "solution.csv",
        np.column_stack((x_mid, u_mid, exact_mid)),
        delimiter=",",
        header="x,y,y_prime,y_second,y_exact",
        comments="",
    )

    print(f"cauchy1: {error_mid:.6e}")
    print(f"cauchy2:      {error_heun:.6e}")
    print(f"Точки сохранены: {output_dir / 'solution.csv'}")


if __name__ == "__main__":
    main()
