import cv2
import numpy as np
import cv2
import itertools

from lsb import LSB


class DCT(LSB):
    def __init__(self, file_path, message = None):
        self.file_path = file_path
        container = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        container = cv2.cvtColor(container, cv2.COLOR_BGR2YCR_CB)
        super().__init__(container, message)
        n, m = self.container.shape[:2]
        print((n, m))
        # self.resized = cv2.resize(self.container,
            # (m + (8 - m % 8), n + (8 - n % 8)))
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
        n, m = self.container.shape[:2]
        # print(matrix[0:8, 0:8])
        blocks = np.array([np.round(matrix[j:j+8, i:i+8]) for (j,i)
            in itertools.product(range(0,n,8), range(0,m,8))])
        # blocks = self._blockshaped(matrix, 8, 8)
        blocks = np.float32(blocks)
        blocks -= 128
        # print(blocks[0])
        blocks = np.array([cv2.dct(block) for block in blocks])
        blocks = np.round(blocks)
        blocks /= self.quant
        blocks = np.round(blocks)
        # blocks = np.round(blocks)
        # print(np.round(blocks))
        return blocks
    
    def _from_dct(self, blocks):
        blocks *= self.quant
        blocks = np.array([np.round(cv2.idct(block)) for block in blocks])
        blocks += 128
        return np.uint8(blocks)

    def _to_elements(self):
        _, blue, _  = cv2.split(self.container)
        blocks = self._to_dct(blue)
        elements = np.int32(blocks[:,0,0].ravel())
        return elements

    def _from_elements(self, elements):
        _ , blue, _ = cv2.split(self.container)
        blocks = self._to_dct(blue)
        blocks[:,0,0] = np.float32(elements) - 255
        blue = self._unblockshaped(self._from_dct(blocks), *self.container.shape[:2])
        self.container[:,:,1] = blue
        # self.container = cv2.resize(self.resized, self.container.shape[:2])
    
    def save(self):
        self.container = cv2.cvtColor(self.container, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(self.file_path, self.container)
    
    def save_as(self, file_path):
        self.container = cv2.cvtColor(self.container, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(file_path, self.container)

    def _blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array looks like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))

    def _unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                   .swapaxes(1,2)
                   .reshape(h, w))


def main():
    with open("Messages/A Sound of Thunder.txt", "rb") as f:
        message = f.read()[:1800]

    # message = ("Hello, world!" * 96).encode()
    # print(message)
    size = len(message)
    jpg = DCT("Images/Lenna.jpg", message)
    jpg.encode()
    jpg.save_as("Images/DCT_Lenna.jpg")
    jpg = DCT("Images/DCT_Lenna.jpg")
    decoded = jpg.decode()[:size]


    print(decoded)

    # print(f"{decoded.decode()}")


if __name__ == "__main__":
    main()