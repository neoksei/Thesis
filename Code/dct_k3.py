import cv2
import numpy as np
import cv2
import itertools
import random

from lsb import LSB


class DCT():
    def __init__(self, file_path, message = None, seed = 0):
        """
        Принимает на вход путь до файла с изображением file_path,
        байтовое сообщение message и попрождающий элемент seed,
        используемый для инициализации ГПСЧ. Возвращает простой
        в использовании JPEG кодер.
        """
        self.file_path = file_path
        # Считываем изображение в BGR
        container = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        # Преобразуем изображение в YCbCr
        self.container = cv2.cvtColor(container, cv2.COLOR_BGR2YCR_CB)

        if message is None:
            self.message = []
        else:
            self.message = message
        
        # Сохраняем seed
        self.seed = seed
        # Используем коэффициенты из средней полосы частот     
        self.coef_table = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 0), (0, 3)]
        # Инициализируем таблицу квантования
        self.quant = np.array([
            [16, 11, 10, 16, 24,  40,  51,  61],
            [12, 12, 14, 19, 26,  58,  60,  55],
            [14, 13, 16, 24, 40,  57,  69,  56],
            [14, 17, 22, 29, 51,  87,  80,  62],
            [18, 22, 37, 56, 68,  109, 103, 77],
            [24 ,35 ,55 ,64 ,81 , 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def _to_dct(self, matrix):
        # Считываем канал и сразу разбиваем его на блоки матриц 8x8
        blocks = self._blockshaped(matrix, 8, 8)
        blocks = np.float32(blocks)
        # Центрируем пиксели
        blocks -= 128
        # К каждому блоку применяем ДКП
        blocks = np.array([cv2.dct(block) for block in blocks])
        # Квантуем каждый блок
        blocks /= self.quant
        # Округляем все элементы
        blocks = np.round(blocks)
        return blocks
    
    def _from_dct(self, blocks):
        # Умножаем каждый элемент на матрицу квантования
        blocks *= self.quant
        # Приминяем обратное ДКП преобразование к каждому блоку
        blocks = np.array([np.round(cv2.idct(block)) for block in blocks])
        # Смещаем пиксели
        blocks += 128
        # Преобразуем каждый пиксель в байт
        return np.uint8(blocks)
    
    def _encode_block(self, block, bit):
        """
        В данном блоке block кодирует bit за счет
        изминения соотношения между тремя псевдослучайными
        элементами.
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self.coef_table, 3)
        # Находим минимум и максимум разницы между третьим
        # и двумя остальными элементами
        M = max(block[k3] - block[k2], block[k3] - block[k1])
        m = min(block[k3] - block[k2], block[k3] - block[k1])
        
        # Кодируем ноль, устанавливая block[k3] минимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантовая коэффициентов
        if bit == False:
            if M < -1:
                if M > -5:
                    block[k3] -= 2
                
                if (block[k2] - block[k3]) < 10:
                    block[k2] += 10 - (block[k2] - block[k3])
                
                if (block[k1] - block[k3]) < 10:
                    block[k1] += 10 - (block[k1] - block[k3])
                
                return block, True

            # Если значение слишком близко к двум другим,
            # мы не трогаем его, чтобы избежать
            # сильных искажений сигнала
            else:
                block[k3] = 0.5 * (block[k2] + block[k1])
                return block, False

        # Кодируем единицу, устанавливая block[k3] максимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантовая коэффициентов
        if bit == True:
            if m > 1:
                if m < 5:
                    block[k3] += 2
                
                if (block[k3] - block[k2]) < 10:
                    block[k2] -= 10 - (block[k3] - block[k2])
                
                if (block[k3] - block[k1]) < 10:
                    block[k1] -= 10 - (block[k3] - block[k1])

                return block, True

            # Если значение слишком близко к двум другим,
            # мы не трогаем его, чтобы избежать
            # сильных искажений сигнала
            else:
                block[k3] = 0.5 * (block[k2] + block[k1])
                return block, False

    def _decode_block(self, block):
        """
        Для данного блока block декодирует
        bit, закодированный с помощью соотношения
        между тремя псевдослучайными элементами
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self.coef_table, 3)
        # Находим минимум и максимум разницы между третьим
        # и двумя остальными элементами
        M = max(block[k3] - block[k2], block[k3] - block[k1])
        m = min(block[k3] - block[k2], block[k3] - block[k1])

        if m < -3 and M < -3:
            return 0
        if m > 3 and M > 3:
            return 1
        # Если коэффициент слишком близко к двум другим,
        # тогда его значение считает неопределенным
        else:
            return -1

    def encode(self):
        """
        Кодирует сообщение в контейнер
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Человек наименее чувствителен к изменению синего цвета,
        # поэтому для кодирования будем использовать синий канал.
        # Алгоритм сжатия JPEG так же использует этот факт,
        # поэтому для повышения робастости сигнала его можно кодировать
        # в канале Y, который отвечает за яркость пикселя
        _, _, blue  = cv2.split(self.container)
        # Преобразуем матрицу в блоки ДКП
        blocks = self._to_dct(blue)
        i = 0
        # Преобразуем сообщение в последовательность бит
        np_message = np.unpackbits(np.frombuffer(self.message, dtype=np.uint8)).ravel()
        for index, block in enumerate(blocks):
            blocks[index], success = self._encode_block(block, np_message[i])
            # Если кодирование завершилось успешно
            if success == True:
                # Переходим к следующему биту
                i += 1
                # И так пока не исчерпаем все сообщение
                if i == len(np_message):
                    break
        # Из блоков ДКП обратно собираем Cb канал
        blue = self._unblockshaped(self._from_dct(blocks), *self.container.shape[:2])
        # Записываем этот канал в изображение
        self.container[:,:,2] = blue

    def decode(self):
        """
        Декодирует сообщение из контейнера
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Извлекаем Cb канал
        _ , _, blue = cv2.split(self.container)
        # Преобразуем канал в блоки ДКП
        blocks = self._to_dct(blue)
        message = []
        for block in blocks:
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
        Перезаписывает исходное сообщение
        """
        self.container = cv2.cvtColor(self.container, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(self.file_path, self.container)
    
    def save_as(self, file_path):
        """
        Сохраняет изображение в файл,
        переданные в параметре file_path
        """
        self.container = cv2.cvtColor(self.container, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(file_path, self.container)

    def _blockshaped(self, arr, nrows, ncols):
        """
        Возвращает массив формы (n, nrows, ncols), где
        n * nrows * ncols = arr.size

        Если массив - это матрица, тогда возвращает массив,
        выглядещий как разбиение этой матрицы на подматрицы.
        """
        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
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
                   .swapaxes(1,2)
                   .reshape(h, w))


def main():
    with open("Messages/A Sound of Thunder.txt", "rb") as f:
        message = f.read()[:100]

    message = ("Hello, stegoworld!").encode()
    size = len(message)
    jpg = DCT("Images/Lenna.jpg", message)
    jpg.encode()
    jpg.save_as("Images/DCT_Lenna.jpg")
    jpg = DCT("Images/DCT_Lenna.jpg")
    decoded = jpg.decode()[:size]

    print(decoded)


if __name__ == "__main__":
    main()