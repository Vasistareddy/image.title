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

args = {'headers': {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.59 Safari/537.36'}, \
'proxies': {'http': '172.20.251.254:20057', 'https': '172.20.251.254:20057'}}

minLineLength = 100
maxLineGap = 50

def lines_extraction(gray: List[int]) -> List[int]:
    """
    this function extracts the lines from the binary image. Cleaning process.
    """
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    return lines

def input(url: str) -> List[int]:
    """
    this function draws the image from the image url given.
    """
    image = None
    try:
        resp = requests.get(url, verify=False, headers=args['headers'], proxies=args['proxies'])
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image

def image2article(url: str) -> Dict[str, str]:
    """
    Core function to extract the title and content from the image.
    """
    title, content = '', ''
    if "http" in url:
        image = input(url)
    else:
        image = cv2.imread(url)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary', im_bw)
    cv2.imwrite('binary.png', im_bw)
    im_bw = ~im_bw
    mask = np.ones(image.shape[:2], dtype="uint8") * 255

    lines = lines_extraction(gray)

    try:
        for line in lines:
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

    im_bw = cv2.bitwise_and(im_bw, im_bw, mask=mask)
    # cv2.imwrite('noise.png', mask)
    # cv2.imshow('mask', mask)
    # cv2.imshow('binary_noise_removal', ~im_bw)
    # cv2.imwrite('binary_noise_removal.png', ~im_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask2 = np.ones(image.shape[:2], dtype="uint8") * 255
    (_, contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)

    # finding the larger text and smearing too
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        if h > 2*avgheight:
            cv2.drawContours(mask2, [c], -1, 0, -1)
            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # for i in range(0, box.shape[0]):
            #     box[i][0] = box[i][0]+20
            # cv2.drawContours(mask2,[box],0,(0,0,255),2)

    x, y = mask2.shape

    value = max(math.ceil(x/100),math.ceil(y/100))+20
    # mask2 = rlsah(x, y, mask2, value)
    mask2 = rlsa.rlsa(mask2, True, False, value) #rlsa application
    (_, contours, _) = cv2.findContours(~mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = [contour for contour in contours if cv2.boundingRect(contour)[2] > 0.60*image.shape[1]]
    mask3 = np.ones(image.shape[:2], dtype="uint8") * 255
    for contour in contours:
        cv2.drawContours(mask3,[contour],0,(0,0,255),2)
    title1 = []
    if contours:
        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            title = title_extract(image, x, y, w, h)
            if title:
                title1.append(title)

    if title1:
        title1.reverse()
        title = ' '.join(title1)
        title = title.replace('\n', ' ')
        title = re.sub(' +',' ', title)
    else:
        title = ''

    content = pytesseract.image_to_string(Image.fromarray(image))
    content = re.sub(' +',' ', content.replace('\n',' ').replace('- ',''))

    if title or content:
        data = {'title': title, 'content': content}
    else:
        data = {'title': '', 'content': ''}
    return data

if __name__ == '__main__':
    url = sys.argv[1]
    data = image2article(url)
    print(data)
