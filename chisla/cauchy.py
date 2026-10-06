import numpy as np


def _prepare(u0, x0, xn, h):
    if not np.isfinite([x0, xn, h]).all():
        raise ValueError("x0, xn и h должны быть конечными числами")
    if h <= 0:
        raise ValueError("Шаг h должен быть положительным")
    initial = np.asarray(u0, dtype=float)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("u0 должен быть непустым одномерным массивом")
    if not np.isfinite(initial).all():
        raise ValueError("u0 должен содержать только конечные числа")

    count = int(np.ceil(abs(xn - x0) / h))
    direction = 1.0 if xn >= x0 else -1.0
    x = x0 + direction * h * np.arange(count + 1)
    x[-1] = xn
    u = np.empty((len(x), initial.size), dtype=float)
    u[0] = initial
    return x, u


def _evaluate(f, x, u):
    value = np.asarray(f(x, u), dtype=float)
    if value.shape != u.shape:
        raise ValueError(
            f"f должна возвращать массив формы {u.shape}, "
            f"получена форма {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("f вернула NaN или бесконечность")
    return value


def cauchy1(f, u0, x0, xn, h):
    x, u = _prepare(u0, x0, xn, h)
    for i in range(len(x) - 1):
        step = x[i + 1] - x[i]
        k1 = _evaluate(f, x[i], u[i])
        k2 = _evaluate(f, x[i] + step / 2, u[i] + step * k1 / 2)
        u[i + 1] = u[i] + step * k2
    return x, u


def cauchy2(f, u0, x0, xn, h):
    x, u = _prepare(u0, x0, xn, h)
    for i in range(len(x) - 1):
        step = x[i + 1] - x[i]
        k1 = _evaluate(f, x[i], u[i])
        k2 = _evaluate(f, x[i + 1], u[i] + step * k1)
        u[i + 1] = u[i] + step * (k1 + k2) / 2
    return x, u
