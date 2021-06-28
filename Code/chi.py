from enum import unique
import jpegio as jio
import scipy.stats
import numpy as np


def chi_attack(file_name: str) -> None:
    """
    Реализует атаку хи-квадрат на JPEG файл. 
    """
    # Считываем ДКП коэффициенты
    dct = jio.read(file_name)
    # Выбираем синий канал, в который спрятано сообщение.
    # Здесь важно, что сообщение представляет собой шум.
    container = dct.coef_arrays[1].ravel()[:]
    # Строим гистограмму
    unique, counts = np.unique(container, return_counts=True)
    hist = dict(zip(unique, counts))
    # Ищем соседние пары
    unique.sort()
    pairs = [(x, y) for (x, y) in zip(unique[:-1], unique[1:]) if x ^ y == 1]
    # Строим наблюдаемое и ожидаемое распределения
    observed = [hist[x] for (x, _) in pairs]
    expected = [(hist[x] + hist[y]) / 2 for (x, y) in pairs]
    # Считаем степень сходства
    _, p = scipy.stats.chisquare(observed, f_exp=expected)
    print(f"{p:.2}")


def main() -> None:
    """
    Проверяет работоспособность программы.
    """
    # Пустой стегоконтейнер
    chi_attack("Images/Lenna.jpg")
    # Заполненный шумом стегоконтейнер
    chi_attack("Images/JSteg_Lenna.jpg")


if __name__ == "__main__":
    main()
