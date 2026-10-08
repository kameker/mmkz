from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path: Path | None, fallback: np.ndarray) -> np.ndarray:
    if path is None:
        return fallback.copy()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def read_required_image(path: Path) -> np.ndarray:
    """Читает обязательный файл и сообщает понятную ошибку, если его нет."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Положите файл {path} в папку проекта")
    return image


def demo_texture(size: int = 480) -> np.ndarray:
    """Текстурное изображение, используемое при отсутствии входного файла."""
    y, x = np.mgrid[:size, :size]
    checker = ((x // 12 + y // 12) % 2) * 80
    waves = 50 * (np.sin(x / 5.5) + np.cos(y / 11.0))
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 24, (size, size))
    b = np.clip(90 + checker + waves + noise, 0, 255)
    g = np.clip(150 + 55 * np.sin((x + y) / 17) + noise, 0, 255)
    r = np.clip(130 + 65 * np.cos(np.hypot(x - 240, y - 240) / 8), 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])


def demo_pair(with_cup: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Две почти одинаковые сцены; во второй можно добавить условную чашку."""
    h, w = 420, 640
    base = np.full((h, w, 3), (160, 180, 195), np.uint8)
    cv2.rectangle(base, (0, 275), (w, h), (70, 105, 130), -1)
    cv2.rectangle(base, (45, 60), (250, 245), (115, 145, 175), -1)
    cv2.circle(base, (510, 115), 65, (80, 155, 210), -1)
    rng = np.random.default_rng(4)
    first = np.clip(base.astype(np.int16) + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
    second = np.clip(base.astype(np.int16) + 3 + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
    if with_cup:
        # Сначала рисуется ручка с заходом под корпус: вся чашка будет одной
        # связной компонентой после бинаризации.
        cv2.ellipse(second, (390, 255), (38, 45), 0, 0, 360, (225, 225, 225), 15)
        cv2.rectangle(second, (275, 205), (390, 325), (225, 225, 225), -1)
        cv2.ellipse(second, (332, 205), (57, 18), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(second, (332, 205), (43, 11), 0, 0, 360, (55, 70, 80), -1)
    return first, second


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def save_grid(images: list[tuple[str, np.ndarray]], path: Path, columns: int = 3,
              normalize: bool = False) -> None:
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis, (title, image) in zip(axes.flat, images):
        shown = image
        if normalize and image.ndim == 2:
            shown = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        axis.imshow(rgb(shown), cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(images):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exercise_1(source: np.ndarray, out: Path) -> None:
    results = [("Оригинал", source)]
    for size in (3, 5, 9, 11):
        results.append((f"Gaussian {size}x{size}", cv2.GaussianBlur(source, (size, size), 0)))

    blur_5_twice = cv2.GaussianBlur(cv2.GaussianBlur(source, (5, 5), 0), (5, 5), 0)
    blur_11 = cv2.GaussianBlur(source, (11, 11), 0)
    delta = cv2.absdiff(blur_5_twice, blur_11)
    delta_visible = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX)
    results += [("5x5 дважды", blur_5_twice), ("11x11 один раз", blur_11),
                ("Абсолютная разность (0..255)", delta),
                ("Разность, контраст усилен", delta_visible)]
    save_grid(results, out / "exercise_1.png")
    cv2.imwrite(str(out / "exercise_1_difference_raw.png"), delta)
    cv2.imwrite(str(out / "exercise_1_difference_visible.png"), delta_visible)
    print(f"1b: средняя абсолютная разность = {delta.mean():.3f}, max = {delta.max()}")


def exercise_2(out: Path) -> None:
    impulse = np.zeros((100, 100), np.float32)
    impulse[50, 50] = 255.0
    blur_5 = cv2.GaussianBlur(impulse, (5, 5), 0)
    blur_9 = cv2.GaussianBlur(impulse, (9, 9), 0)
    blur_5_twice = cv2.GaussianBlur(blur_5, (5, 5), 0)
    delta = cv2.absdiff(blur_5_twice, blur_9)
    save_grid([
        ("Импульс (нормированный показ)", impulse),
        ("Gaussian 5x5", blur_5),
        ("Gaussian 9x9", blur_9),
        ("5x5 дважды", blur_5_twice),
        ("Разность", delta),
    ], out / "exercise_2.png", normalize=True)

    report = (
        "Центральные фрагменты (реальные значения яркости):\n"
        f"5x5:\n{np.array2string(blur_5[47:54, 47:54], precision=2)}\n\n"
        f"9x9:\n{np.array2string(blur_9[45:56, 45:56], precision=2)}\n\n"
        f"5x5 дважды:\n{np.array2string(blur_5_twice[45:56, 45:56], precision=2)}\n"
        f"Средняя абсолютная разность 9x9 и (5x5 дважды): {delta.mean():.6f}\n"
    )
    (out / "exercise_2_values.txt").write_text(report, encoding="utf-8")
    print("2: импульс превратился в двумерное ядро Гаусса; значения записаны в exercise_2_values.txt")


def exercise_3(source: np.ndarray, out: Path) -> None:
    fixed = [("Оригинал", source)]
    auto = [("Оригинал", source)]
    for sigma in (1, 4, 6):
        fixed.append((f"9x9, sigma={sigma}", cv2.GaussianBlur(source, (9, 9), sigmaX=sigma)))
        auto.append((f"auto, sigma={sigma}", cv2.GaussianBlur(source, (0, 0), sigmaX=sigma)))
    save_grid(fixed, out / "exercise_3a_fixed_kernel.png", columns=2)
    save_grid(auto, out / "exercise_3b_auto_kernel.png", columns=2)

    horizontal = cv2.GaussianBlur(source, (0, 0), sigmaX=1, sigmaY=9)
    vertical = cv2.GaussianBlur(source, (0, 0), sigmaX=9, sigmaY=1)
    both_orders_1 = cv2.GaussianBlur(horizontal, (0, 0), sigmaX=9, sigmaY=1)
    both_orders_2 = cv2.GaussianBlur(vertical, (0, 0), sigmaX=1, sigmaY=9)
    kernel_9 = cv2.GaussianBlur(source, (9, 9), sigmaX=0, sigmaY=0)
    save_grid([
        ("3c: sigmaX=1, sigmaY=9", horizontal),
        ("3d: sigmaX=9, sigmaY=1", vertical),
        ("3e: c затем d", both_orders_1),
        ("3e: d затем c", both_orders_2),
        ("3f: 9x9, sigma auto", kernel_9),
        ("Разность двух порядков", cv2.absdiff(both_orders_1, both_orders_2)),
    ], out / "exercise_3c_f.png", columns=2)
    print("3e: MAE между двумя порядками =", f"{cv2.absdiff(both_orders_1, both_orders_2).mean():.4f}")
    print("3f: MAE между последовательным размытием и 9x9 =",
          f"{cv2.absdiff(both_orders_1, kernel_9).mean():.4f}")


def exercise_4(src1: np.ndarray, src2: np.ndarray, out: Path) -> None:
    if src1.shape != src2.shape:
        raise ValueError("Для упражнения 4 изображения должны иметь одинаковый размер")
    diff12 = cv2.absdiff(src1, src2)  # В условии src1-src1 — очевидная опечатка.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleandiff = cv2.dilate(cv2.erode(diff12, kernel), kernel)  # opening
    dirtydiff = cv2.erode(cv2.dilate(diff12, kernel), kernel)  # closing
    save_grid([("src1", src1), ("src2", src2), ("diff12", diff12),
               ("cleandiff: erode -> dilate", cleandiff),
               ("dirtydiff: dilate -> erode", dirtydiff)], out / "exercise_4.png")


def exercise_5(background: np.ndarray, cup_scene: np.ndarray, out: Path,
               threshold: int) -> np.ndarray:
    if background.shape != cup_scene.shape:
        raise ValueError("Для упражнения 5 изображения должны иметь одинаковый размер")
    gray1 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cup_scene, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(opened, np.ones((3, 3), np.uint8))
    contour = cv2.bitwise_xor(opened, eroded)
    save_grid([("Фон", gray1), ("Сцена с чашкой", gray2), ("absdiff", diff),
               (f"Бинаризация, T={threshold}", binary), ("MORPH_OPEN", opened),
               ("Контур: mask XOR erode(mask)", contour)], out / "exercise_5.png", columns=3)
    cv2.imwrite(str(out / "cup_contour.png"), contour)
    return opened


def keep_largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Ожидается одноканальная маска типа uint8")

    work = mask.copy()
    height, width = work.shape
    largest_seed: tuple[int, int] | None = None
    largest_area = 0
    component_count = 0

    # Указатель начинает в левом верхнем углу и идет по строкам изображения.
    for y in range(height):
        for x in range(width):
            if work[y, x] != 255:
                continue

            component_count += 1
            seed = (x, y)  # floodFill принимает координаты в порядке (x, y)
            area, _, _, _ = cv2.floodFill(work, None, seed, 128, flags=8)

            if area > largest_area:
                # Найдена более крупная область: удаляем прежнего лидера.
                if largest_seed is not None:
                    cv2.floodFill(work, None, largest_seed, 0, flags=8)
                largest_seed = seed
                largest_area = area
            else:
                # Текущая компонента меньше уже запомненной.
                cv2.floodFill(work, None, seed, 0, flags=8)

    result = np.zeros_like(work)
    if largest_seed is not None:
        # В work осталась единственная область со значением 128.
        cv2.floodFill(work, None, largest_seed, 255, flags=8)
        result[work == 255] = 255

    return result, largest_area, component_count


def exercise_8_9(opened_mask: np.ndarray, out: Path) -> None:
    largest, area, count = keep_largest_component(opened_mask)
    save_grid([
        ("Исходная очищенная маска", opened_mask),
        (f"Самая большая компонента, S={area}", largest),
    ], out / "exercise_8_9.png", columns=2)
    cv2.imwrite(str(out / "cup_largest_component.png"), largest)
    print(f"8-9: найдено компонент: {count}; площадь наибольшей: {area} пикселей")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture", type=Path, help="Изображение для упражнений 1 и 3")
    parser.add_argument("--threshold", type=int, default=25, help="Порог маски чашки (0..255)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Каталог результатов")
    parser.add_argument("--show", action="store_true", help="После вычислений показать PNG на экране")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold должен быть в диапазоне 0..255")
    args.output.mkdir(parents=True, exist_ok=True)

    texture = read_image(args.texture, demo_texture())
    # Упражнение 4 использует два фиксированных файла из папки проекта.
    scene1 = read_required_image(Path("i3.jpg"))
    scene2 = read_required_image(Path("i4.jpg"))
    # Упражнение 5: i1.jpg — первый снимок без чашки,
    # i2.jpg — второй снимок той же сцены с чашкой.
    first_photo = read_required_image(Path("i1.jpg"))
    second_photo = read_required_image(Path("i2.jpg"))

    exercise_1(texture, args.output)
    exercise_2(args.output)
    exercise_3(texture, args.output)
    exercise_4(scene1, scene2, args.output)
    opened_mask = exercise_5(first_photo, second_photo, args.output, args.threshold)
    exercise_8_9(opened_mask, args.output)
    print(f"Готово. Результаты: {args.output.resolve()}")

    if args.show:
        for file in sorted(args.output.glob("exercise_*.png")):
            image = cv2.imread(str(file))
            cv2.imshow(file.stem, image)
        print("Нажмите любую клавишу в окне изображения для выхода")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
