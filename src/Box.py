import numpy as np
from PIL import Image

class Box():
    def __init__(self, im, pos, size):
        self.img: Image.Image = im
        self.arr: np.array = None
        self.pos = pos
        self.size = size
        self.digit = 0