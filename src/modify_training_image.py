import cv2
import numpy as np
from PIL import Image

def modify(img, blur=4, ): # input image as numpy array
    img = cv2.blur(img, blur)
    