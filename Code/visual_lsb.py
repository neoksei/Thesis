import numpy as np
import cv2


def main() -> None:
    """
    Проверяет работоспособность программы.
    """
    # Считаем синий канал оригинала
    blue_original = cv2.imread("Images/Lenna.png", 0)
    # Считаем синий канал модифицированного сообщения
    blue_stego = cv2.imread("Images/LSB_Lenna.png", 0)
    # Получим только последний бит.
    # Для контрастности умножим полученное значение на 255,
    # таким образом в матрицу будут лишь значения 0 и 255.
    bw_original = (blue_original & 1) * 255
    bw_stego = (blue_stego & 1) * 255
    # Сохраним полученные изображения в градации серого.
    cv2.imwrite("Images/BW_Lenna.png", bw_original)
    cv2.imwrite("Images/BW_LSB_Lenna.png", bw_stego)


if __name__ == "__main__":
    main()
