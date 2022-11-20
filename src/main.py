"""
The Neural Network is racist and only 
correctly predicts numbers 
that are white on a black background
"""

# %%
import copy
import sys
from PIL import Image, ImageFont, ImageDraw
import os
import cv2
import numpy as np

from DigitRecognizer import DigitRecognizer
from Convert import ImageConverter
# %%
Recognizer = DigitRecognizer()
Converter = ImageConverter()

boxes = Converter.convert('22.jpg')

# %%
# for i in range(len(boxes)):
#     res = Recognizer.recognize(boxes[i].img)
#     boxes[i].digit = np.argmax(res)
#     print(boxes[i].digit)

res = Recognizer.recognize(boxes[3].img)

print(np.argmax(res), np.max(res))


# %%
