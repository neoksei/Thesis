from hashlib import new
from os import nice
from typing import Container
import cv2
import numpy as np
import jpegio as jio
import cv2
import itertools
import random

from numpy.core.shape_base import block


class DCT():
    def __init__(self, file_name, message=None, seed=0):
        """
        Принимает на вход путь до файла с изображением file_path,
        байтовое сообщение message и попрождающий элемент seed,
        используемый для инициализации ГПСЧ. Возвращает простой
        в использовании JPEG кодер.
        """
        self.file_name = file_name
        # Считываем ДКП коэффициенты
        self.dct = jio.read(self.file_name)
        # Считываем канал яркости.
        self.container = self.dct.coef_arrays[0]

        if message is None:
            self.message = []
        else:
            self.message = message

        # Сохраняем seed
        self.seed = seed
        # Коэффициенты для встраивания
        self.stego_coef = [(i, j) for i in range(8) for j in range(8)
                           if i + j < 5 and (i, j) != (0, 0)]

        # Высокочастотные коэффициенты
        self.high_coef = np.array(
            [i + j > 9 for i in range(8) for j in range(8)]
        ).reshape(8, 8)

        # Низкочастотные коэффициенты
        self.low_coef = np.array(
            [i + j < 5 for i in range(8) for j in range(8)]
        ).reshape(8, 8)

        # Сохраняем порог различения
        self.P = 3
        # Сохраняем порог яркости
        self.Pl = 210
        # Сохраняем порог монотонности
        self.Ph = 40
        self.x = 0

    def _check_block(self, block):
        # Проверяем блок на порог яркости и монотонности
        l = np.absolute(block[self.low_coef].sum())
        h = np.absolute(block[self.high_coef].sum())
        # return True
        return l >= self.Pl and h <= self.Ph

    def _encode_block(self, block, bit):
        """
        В данном блоке block кодирует bit за счет
        изминения соотношения между тремя псевдослучайными
        элементами.
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self.stego_coef, 3)

        # Кодируем ноль, устанавливая block[k3] минимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантования коэффициентов
        if bit == False:
            m = min(block[k1], block[k2])
            block[k3] = m - self.P / 2

            if block[k1] == m:
                block[k1] += self.P / 2
            else:
                block[k2] += self.P / 2

        # Кодируем единицу, устанавливая block[k3] максимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантовая коэффициентов
        else:
            m = max(block[k1], block[k2])
            block[k3] = m + self.P / 2

            if block[k1] == m:
                block[k1] -= self.P / 2
            else:
                block[k2] -= self.P / 2

        return block

    def _decode_block(self, block):
        """
        Для данного блока block декодирует
        bit, закодированный с помощью соотношения
        между тремя псевдослучайными элементами
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self.stego_coef, 3)

        # Находим минимум и максимум разницы между третьим
        # и двумя остальными элементами
        M = max(block[k1], block[k2], block[k3])
        m = min(block[k1], block[k2], block[k3])

        if block[k3] == M:
            return 1

        if block[k3] == m:
            return 0
        else:
            return -1

    def encode(self):
        """
        Кодирует сообщение в контейнер
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Разбиваем массив ДКП коэффициентов на блоки
        blocks = self._blockshaped(self.container, 8, 8)
        # Находим положение подходящих блоков
        mask = np.array([self._check_block(block) for block in blocks])
        # В подходящих блоках меняем соотношения коэффициентов
        nice_blocks = blocks[mask]
        print(len(nice_blocks))
        # Преобразуем сообщение в бинарный вид
        np_message = np.unpackbits(np.frombuffer(
            self.message, dtype=np.uint8)).ravel()
        # Находим длину сообщения
        n = len(np_message)
        # Кодируем сообщение
        new_nice_blocks = []
        for i in range(n):
            new_nice_blocks.append(self._encode_block(
                nice_blocks[i], np_message[i]))
        # Перезаписываем подходящие блоки
        nice_blocks[:n] = np.array(new_nice_blocks)
        blocks[mask] = nice_blocks
        # Соединяем блоки обратно в контейнер
        self.container = self._unblockshaped(blocks, *self.container.shape)
        # Сохраняем информацию в изображение
        self.dct.coef_arrays[0].ravel()[:] = self.container.ravel()
        self.x = 0

    def decode(self):
        """
        Декодирует сообщение из контейнера
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Разбиваем массив ДКП коэффициентов на блоки
        blocks = self._blockshaped(self.container, 8, 8)
        message = []
        # Находим позиции подходящих блоков
        mask = np.array([self._check_block(block) for block in blocks])
        # Находим подходящие блоки
        nice_blocks = blocks[mask]
        print(len(nice_blocks))
        # Декодируем сообщение
        for block in nice_blocks:
            # Пробуем декодировать бит
            bit = self._decode_block(block)
            # -1 сигнализирует о том,
            # что декодирование не удалось
            if bit != -1:
                message.append(bit)
        # Из бит собираем исходное сообщение
        message = np.packbits(message)
        # Преобразуем его в байты
        return message.tobytes()

    def save(self):
        """
        Перезаписывает исходный файл
        новый контейнером.
        """
        jio.write(self.dct, self.file_name)

    def save_as(self, file_name):
        """
        Сохраняет контейнер в файл,
        заданный параментром file_name.
        """
        jio.write(self.dct, file_name)

    def _blockshaped(self, arr, nrows, ncols):
        """
        Возвращает массив формы (n, nrows, ncols), где
        n * nrows * ncols = arr.size

        Если массив - это матрица, тогда возвращает массив,
        выглядещий как разбиение этой матрицы на подматрицы.
        """
        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1, 2)
                   .reshape(-1, nrows, ncols))

    def _unblockshaped(self, arr, h, w):
        """
        Возвращает матрицу формы (h, w), где
        h * w = arr.size

        Если матрица формы (n, nrows, ncols), где n - это подматрицы
        формы (nrows, ncols), тогда возвращает матрицу, составленную
        из этих подматриц.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                   .swapaxes(1, 2)
                   .reshape(h, w))


def main():
    with open("Messages/A Sound of Thunder.txt", "rb") as f:
        message = f.read()[:100]

    message = ("Hello").encode()
    size = len(message)
    jpg = DCT("Images/Lenna.jpg", message)
    jpg.encode()
    jpg.save_as("Images/DCT_Lenna.jpg")
    jpg = DCT("Images/DCT_Lenna.jpg")
    decoded = jpg.decode()[:size]

    print(decoded)


if __name__ == "__main__":
    main()
