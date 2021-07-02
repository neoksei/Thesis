import numpy as np
import jpegio as jio
import random


class BMYY():
    """
    Реализация метода Бенгама-Мемона-Эо-Юнга.
    """

    def __init__(self, file_name: str, message: bytes = None,
                 seed: int = 0) -> None:
        """
        Принимает на вход путь до файла с изображением file_path,
        байтовое сообщение message и попрождающий элемент seed,
        используемый для инициализации ГПСЧ. Возвращает простой
        в использовании JPEG кодер.
        """
        self._file_name = file_name
        # Считываем ДКП коэффициенты
        self._dct = jio.read(self._file_name)
        # Считываем канал яркости.
        self._container = self._dct.coef_arrays[0]

        if message is None:
            self.message = []

        else:
            self.message = message

        # Сохраняем seed
        self.seed = seed
        # Коэффициенты для встраивания
        self._stego_coef = [(i, j) for i in range(8) for j in range(8)
                            if i + j < 5 and (i, j) != (0, 0)]

        # Высокочастотные коэффициенты
        self._high_coef = np.array(
            [i + j > 9 for i in range(8) for j in range(8)]
        ).reshape(8, 8)

        # Низкочастотные коэффициенты
        self._low_coef = np.array(
            [i + j < 5 for i in range(8) for j in range(8)]
        ).reshape(8, 8)

        # Сохраняем порог различения
        self._P = 3
        # Сохраняем порог яркости
        self._Pl = 210
        # Сохраняем порог монотонности
        self._Ph = 40

    def _is_suitable_block(self, block: np.array) -> bool:
        # Проверяем блок на порог яркости и монотонности
        l = np.absolute(block[self._low_coef].sum())
        h = np.absolute(block[self._high_coef].sum())
        # return True
        return l >= self._Pl and h <= self._Ph

    def _encode_block(self, block: np.array, bit: bool) -> np.array:
        """
        В данном блоке block кодирует bit за счет
        изминения соотношения между тремя псевдослучайными
        элементами.
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self._stego_coef, 3)

        # Кодируем ноль, устанавливая block[k3] минимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантования коэффициентов
        if bit == False:
            m = min(block[k1], block[k2])
            block[k3] = m - self._P / 2

            if block[k1] == m:
                block[k1] += self._P / 2

            else:
                block[k2] += self._P / 2

        # Кодируем единицу, устанавливая block[k3] максимальным
        # из трех элементов так, чтобы это соотношение
        # сохранилось после квантовая коэффициентов
        else:
            m = max(block[k1], block[k2])
            block[k3] = m + self._P / 2

            if block[k1] == m:
                block[k1] -= self._P / 2

            else:
                block[k2] -= self._P / 2

        return block

    def _decode_block(self, block: np.array) -> bool:
        """
        Для данного блока block декодирует
        bit, закодированный с помощью соотношения
        между тремя псевдослучайными элементами
        """
        # С помощью ГПСЧ выбираем случайные элементы блока
        k1, k2, k3 = random.sample(self._stego_coef, 3)
        # Находим максимум разницы между третьим
        # и двумя остальными элементами
        M = max(block[k1], block[k2], block[k3])

        if block[k3] == M:
            return True

        else:
            return False

    def _blockshaped(self, arr: np.array, nrows: int, ncols: int) -> np.array:
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

    def _unblockshaped(self, arr: np.array, h: int, w: int) -> np.array:
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

    def encode(self) -> bytes:
        """
        Кодирует сообщение в контейнер и возвращает позиции подходящих блоков.
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Разбиваем массив ДКП коэффициентов на блоки
        blocks = self._blockshaped(self._container, 8, 8)
        # Находим положение подходящих блоков
        mask = np.array([self._is_suitable_block(block) for block in blocks])
        # В подходящих блоках меняем соотношения коэффициентов
        suitable_blocks = blocks[mask]
        # Преобразуем сообщение в бинарный вид
        np_message = np.unpackbits(np.frombuffer(
            self.message, dtype=np.uint8)).ravel()
        # Находим длину сообщения
        n = len(np_message)
        # Кодируем сообщение
        encoded_blocks = []

        for i in range(n):
            encoded_blocks.append(self._encode_block(
                suitable_blocks[i], np_message[i]))

        # Перезаписываем подходящие блоки
        suitable_blocks[:n] = np.array(encoded_blocks)
        blocks[mask] = suitable_blocks
        # Соединяем блоки обратно в контейнер
        self._container = self._unblockshaped(blocks, *self._container.shape)
        # Сохраняем информацию в изображение
        self._dct.coef_arrays[0].ravel()[:] = self._container.ravel()
        # Возвращаем позиции встраивания
        return np.packbits(mask).tobytes()

    def decode(self, positions: bytes) -> bytes:
        """
        Декодирует сообщение из контейнера
        """
        # Инициализируем ГПСЧ
        random.seed(self.seed)
        # Разбиваем массив ДКП коэффициентов на блоки
        blocks = self._blockshaped(self._container, 8, 8)
        message = []
        # Находим позиции подходящих блоков
        mask = np.unpackbits(np.frombuffer(positions, np.uint8)).astype(bool)
        # Находим подходящие блоки
        suitable_blocks = blocks[mask[:len(blocks)]]

        # Декодируем сообщение
        for block in suitable_blocks:
            # Декодируем бит
            bit = self._decode_block(block)
            message.append(bit)

        # Из бит собираем исходное сообщение
        message = np.packbits(message)
        # Преобразуем его в байты
        return message.tobytes()

    def save(self) -> None:
        """
        Перезаписывает исходный файл
        новый контейнером.
        """
        jio.write(self._dct, self._file_name)

    def save_as(self, file_name: str) -> None:
        """
        Сохраняет контейнер в файл,
        заданный параментром file_name.
        """
        jio.write(self._dct, file_name)


def main() -> None:
    """
    Проверяет работоспособность алгоритма.
    """
    # Переводим сообщение в байтовую форму
    message = ("Hello, stegoworld!").encode()
    # Запоминаем размер бинарного сообщения
    size = len(message)
    # Создаем кодер
    jpg = BMYY("Images/Lenna.jpg", message)
    # Скрываем сообщение, запоминаем позиции
    # блоков, в которые встроены биты
    positions = jpg.encode()
    jpg.save_as("Images/BMYY_Lenna.jpg")
    jpg = BMYY("Images/BMYY_Lenna.jpg")
    decoded = jpg.decode(positions)[:size]
    # Убеждаемся, что метод работает
    print(decoded)


if __name__ == "__main__":
    main()
