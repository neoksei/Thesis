import cv2
import numpy as np
import cv2


class DCT():
    def __init__(self, file_path, message):
        self.image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        self.quntitization_table = np.array([
            [16, 11, 10, 16, 24,  40,  51,  61],
            [12, 12, 14, 19, 26,  58,  60,  55],
            [14, 13, 16, 24, 40,  57,  69,  56],
            [14, 17, 22, 29, 51,  87,  80,  62],
            [18, 22, 37, 56, 68,  109, 103, 77],
            [24 ,35 ,55 ,64 ,81 , 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def encode(self):
        elements = self._to_elements()
        np_message = np.unpackbits(np.frombuffer(self.message, dtype=np.uint8)).ravel()
        rows_n, columns_n = self.image.shape[:2]
        self.image = cv2.resize(self.image,
            (columns_n + (8 - columns_n % 8), rows_n + (8 - rows_n % 8))
            )
        