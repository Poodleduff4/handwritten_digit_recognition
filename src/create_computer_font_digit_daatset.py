from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import random
from modify_training_image import modify

def create_dataset(num_images, font, size):
    for i in range(num_images):
        img = np.array((size))
        img = cv2.addText(img, random.randint(0, 10)) # for only one font, can make for different fonts with random font name in list of names
        img = modify(img)
