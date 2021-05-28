from lsb import LSB
from PIL import Image

import numpy as np


class PNG(LSB):

    def __init__(self, file_name: str, message: str = None):
        """
        Возвращает простой PNG кодер,
        принимает на вход имя файла и сообщение.
        """
        self.file_name = file_name
        container = np.array(Image.open(file_name))
        super().__init__(container, message)

    def _to_elements(self):
        """
        Возвращает репрезентацию контейнера
        как последовательности элементов.
        """
        return self.container.ravel()[:]

    def _from_elements(self, elements):
        """
        Строит контейнер по последовательности
        элементов.
        """
        self.container.ravel()[:] = elements

    def save(self):
        """
        Перезаписывает исходный файл
        новый контейнером.
        """
        image = Image.fromarray(self.container)
        image.save(self.file_name)

    def save_as(self, file_name):
        """
        Сохраняет контейнер в файл,
        заданный параментром file_name.
        """
        image = Image.fromarray(self.container)
        image.save(file_name)


def main():
    # Считываем сообщение.
    with open("Messages/Alice in wonderland.txt", "rb") as f:
        message = f.read()

    # Запоминаем длину сообщения.
    size = len(message)
    # Кодируем сообщение.
    png = PNG("Images/Lenna.png", message)
    png.encode()
    png.save_as("Images/LSB_Lenna.png")
    # Переоткрываем изображение.
    png = PNG("Images/LSB_Lenna.png")
    # Декодируем сообщение.
    decoded = png.decode()

    # Проверяем, что сообщения до и после совпадают.
    new_message = decoded[:size]
    print(f"{new_message == message}")


if __name__ == "__main__":
    main()
