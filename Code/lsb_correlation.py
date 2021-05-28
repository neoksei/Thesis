import numpy as np
import cv2

# Считаем синий канал оригинала.
blue_original = cv2.imread("Images/Lenna.png", 0)

# Получим только последний бит.
bin_original = (blue_original & 1)

# Преобразуем матрицу бит в массив бит.
bin_original = bin_original.ravel()[:]

# Для сравнения сгенерируем псевдослучайную
# равномерно распределнную последовательность
# бит, чтобы смоделировать стегосообщение.
bin_stego = np.random.randint(2, size=len(bin_original))

# Сравним 2 соседних бита у оригинального изображения
# и промоделированного.
original_comparison = (bin_original[1:] == bin_original[:-1])
stego_comparison = (bin_stego[1:] == bin_stego[:-1])

# Напишем функцию, печатающую статистику по нашему распределению
def pretty_stat(X, name):
    size = len(X)
    # Посчитаем частоты элементов последовательности
    comparison = dict(zip(*np.unique(X, return_counts=True)))
    # Выведем статистику на экран
    stat = f"{name}: "
    for key, value in comparison.items():
        stat += f"P({key}) = {value / size:.2}, "
    
    print(stat[:-2])

# Посмотрим на получившееся распределение
pretty_stat(original_comparison, "Original")
pretty_stat(stego_comparison, "Stego")
