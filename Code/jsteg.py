import jpegio as jio
import numpy as np
import cv2
from lsb import LSB


class JSteg(LSB):
    """
    Реализация стеганографического алгоритма JSteg.
    """

    def __init__(self, file_name: str, message: str = None) -> None:
        """
        Возвращает простой JSteg кодер,
        принимает на вход имя файла и сообщение.
        """
        self.file_name = file_name
        # Считываем все коэффициенты ДКП
        self.dct = jio.read(self.file_name)
        # Оставляем только Cb канал.
        # Коэффициенты упорядочены в зигзагообразном порядке
        container = self.dct.coef_arrays[1]
        super().__init__(container, message)

    def _to_elements(self) -> np.array:
        """
        Возвращает репрезентацию контейнера
        как последовательности элементов.
        """
        return self.container.ravel()[:]

    def _from_elements(self, elements: np.array) -> None:
        """
        Строит контейнер по последовательности
        элементов.
        """
        self.dct.coef_arrays[1].ravel()[:] = elements

    def save(self) -> None:
        """
        Перезаписывает исходный файл
        новый контейнером.
        """
        jio.write(self.dct, self.file_name)

    def save_as(self, file_name: str) -> None:
        """
        Сохраняет контейнер в файл,
        заданный параментром file_name.
        """
        jio.write(self.dct, file_name)


def main() -> None:
    """
    Проверяет работоспособность программы.
    """
    # Считываем сообщение.
    with open("Messages/Alice in wonderland.txt", "rb") as f:
        message = f.read()[:80000]

    # Запоминаем длину сообщения.
    size = len(message)
    # Кодируем сообщение.
    jsteg = JSteg("Images/Lenna.jpg", message)
    jsteg.encode()
    jsteg.save_as("Images/JSteg_Lenna.jpg")
    # Переоткрываем изображение.
    jsteg = JSteg("Images/JSteg_Lenna.jpg")
    # Декодируем сообщение.
    decoded = jsteg.decode()

    # Проверяем, что сообщения до и после совпадают.
    new_message = decoded[:size].decode()
    print(f"{new_message == message.decode()}")


if __name__ == "__main__":
    main()
