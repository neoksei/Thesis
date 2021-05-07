from abc import ABC, abstractmethod
from bitarray import bitarray

import numpy as np


class LSB(ABC):
    def __init__(self, container: np.array, message = None):
        if message is None:
            self.message = []
        else:
            self.message = message

        self.container = container

    def encode(self):
        elements = self._to_elements()
        np_message = np.unpackbits(np.frombuffer(self.message, dtype=np.uint8)).ravel()
        n = len(np_message)
        elements[:n] = (elements[:n] & ~1) | np_message
        self._from_elements(elements)

    def decode(self):
        elements = self._to_elements()
        np_message = (elements & 1)
        message = np.packbits(np_message.reshape(-1, 8), axis=-1).tobytes()
        return message

    @abstractmethod
    def _to_elements(self):
        pass

    @abstractmethod
    def _from_elements(self, elements):
        pass
