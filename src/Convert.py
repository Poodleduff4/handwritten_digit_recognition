# %%
from PIL import Image
import cv2
import numpy as np
from Box import Box

class ImageConverter():

    def __init__(self):
        print("Image Converter")

    def convert(self, filename):
        boxes = []


        # im = cv2.imread('/Users/lukeguardino/Documents/Python crap/Sudoku/input_files/'+filename)
        im = cv2.imread('/Users/lukeguardino/Documents/handwritten-digit-recognition/input_files/'+filename)
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im_blank = np.zeros(im.shape)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        # Blur the image
        blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        edges = cv2.Canny(thresh, 180, 220)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 10, 20)
        lines_img = np.ones(im.shape[:2], dtype="uint8")*255
        try: # no boxes, probably only one number on blank background
            for line in lines:
                # print(line[0])
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1,y1), (x2,y2), (0,0,0), 1)


            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][1:]
            mask = np.ones(im.shape[:2], dtype="uint8")*255

            for c in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[c])
                # if w*h > 1000:
                if abs((w/h)-1) < 0.10: # percentage of error in width to height ratio
                    cv2.rectangle(mask, (x,y), (x+w,y+h), (0,0,255), -1)
                    crop_img = Image.fromarray(im[y:y+h, x:x+w])
                    boxes.append(Box(crop_img, (x,y), (w,h)))
                    cv2.drawContours(im_blank, contours, c, (0,255,0),3)

            res_final = cv2.bitwise_and(im, im, mask=cv2.bitwise_not(mask))

        except:
            boxes.append(Box(im, 0,0))
        # cv2.imshow('original', thresh)
        # cv2.imshow('edges', edges)
        # cv2.imshow('contours', im_blank)
        # cv2.imshow('final', res_final)
        # cv2.waitKey(0)

        return boxes
# %%
