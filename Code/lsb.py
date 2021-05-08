from abc import ABC, abstractmethod
from bitarray import bitarray

import numpy as np


class LSB(ABC):
    def __init__(self, container: np.array, message = None):
        """
        Возвращает простой lsb-кодер,
        принимает на вход контейнер и сообщение (массив байт).
        """
        if message is None:
            # По умолчанию сообщение пустое
            self.message = []
        else:
            self.message = message

        self.container = container

    def encode(self):
        """
        Кодирует сообщение в контейнер
        """
        # Получаем последовательность элементов контейнера
        elements = self._to_elements()
        # Преобразуем сообщение к бинарному виду
        np_message = np.unpackbits(np.frombuffer(self.message, dtype=np.uint8)).ravel()
        n = len(np_message)
        # Меняем наименее значимый бит так,
        # чтобы он кодировал биты сообщения
        elements[:n] = (elements[:n] & ~1) | np_message
        # Из элементов собираем контейнер обратно
        self._from_elements(elements)

    def decode(self):
        """
        Декодирует сообщение из контейнера
        """
        # Получаем последовательность элементов контейнера
        elements = self._to_elements()
        # Выбираем размер сообщения так, чтобы он был кратен размеру байта
        size = len(elements) // 8 * 8
        # Сообщение считываем из наименее значащих бит элементов контейнера
        np_message = (elements[:size] & 1)
        # Преобразуем битовую последовательность в байты
        message = np.packbits(np_message.reshape(-1, 8), axis=-1).tobytes()
        return message

    @abstractmethod
    def _to_elements(self):
        """
        Преобразует контейнер в последовательность элементов
        """
        pass

    @abstractmethod
    def _from_elements(self, elements):
        """
        Собирает контейнер из элементов
        """
        pass
