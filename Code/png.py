from lsb import LSB
from PIL import Image

import numpy as np


class PNG(LSB):

    def __init__(self, file_name: str, message: str):
        self.file_name = file_name
        container = np.array(Image.open(file_name))
        super().__init__(container, message)

    def _to_elements(self):
        return self.container.ravel()[:]

    def _from_elements(self, elements):
        self.container.ravel()[:] = elements

    def save(self):
        image = Image.fromarray(self.container)
        image.save(self.file_name)

    def save_as(self, file_name):
        image = Image.fromarray(self.container)
        image.save(file_name)


def main():
    with open("Messages/Alice in wonderland.txt", "rb") as f:
        message = f.read()

    size = len(message)
    png = PNG("Images/Lenna.png", message)
    png.encode()
    png.save_as("Images/LSB_Lenna.png")
    decoded = png.decode()

    new_message = decoded[:size].decode()
    print(f"{new_message == message.decode()}")


if __name__ == "__main__":
    main()
