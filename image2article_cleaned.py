from typing import List, Dict
import cv2
import numpy as np
import os
import math
import glob
import pytesseract
from PIL import Image
import sys
import requests
import re
from pythonRLSA import rlsa

minLineLength = 100
maxLineGap = 50

def lines_extraction(gray: List[int]) -> List[int]:
    """
    this function extracts the lines from the binary image. Cleaning process.
    """
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    return lines

image = cv2.imread('images_work_directory/image.png') #reading the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
(thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image
im_bw = ~im_bw

mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
lines = lines_extraction(gray) # line extraction

try:
    for line in lines:
        """
        drawing extracted lines on mask
        """
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 3)
except TypeError:
    pass
(_, contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
avgArea = sum(areas)/len(areas)
for c in contours:
    if cv2.contourArea(c)>60*avgArea:
        cv2.drawContours(mask, [c], -1, 0, -1)

im_bw = cv2.bitwise_and(im_bw, im_bw, mask=mask) # nullifying the mask over binary

mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create the blank image
(_, contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
heights = [cv2.boundingRect(contour)[3] for contour in contours]
avgheight = sum(heights)/len(heights)

# finding the larger text
for c in contours:
    [x,y,w,h] = cv2.boundingRect(c)
    if h > 2*avgheight:
        cv2.drawContours(mask, [c], -1, 0, -1)

cv2.imshow('mask', mask)

x, y = mask.shape # image dimensions

value = max(math.ceil(x/100),math.ceil(y/100))+20
mask = rlsa.rlsa(mask, True, False, value) #rlsa application

cv2.imshow('mask1', mask)

(_, contours, _) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if w > 0.60*image.shape[1]:# width heuristic applied
        title = image[y: y+h, x: x+w] 
        mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image

title = pytesseract.image_to_string(Image.fromarray(mask2))
content = pytesseract.image_to_string(Image.fromarray(image))

print('title - {0}, content - {1}'.format(title, content))

cv2.imshow('title', mask2)
# cv2.imwrite('title.png', mask2)
cv2.imshow('content', image)
# cv2.imshow('content.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()