import os
import cv2
import math
import numpy as np

if __name__ == '__main__':
    name = 'darts_with_arrow_segmentation'
    path = 'output/probability/' + name + '.jpg'
    img = cv2.imread(path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', img1)

    # img1 = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=11)
    # img1 = cv2.GaussianBlur(img1, (3, 3), 2)
    # img1 = cv2.Laplacian(img1, cv2.CV_8U)
    img1 = cv2.medianBlur(img1, 5)
    img1 = cv2.Canny(img1, 30, 200)
    cv2.imshow('blur', img1)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     if cv2.contourArea(contour) < 150:
    #         cv2.drawContours(img, contour, -1, (0, 255, 0), 1)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    cv2.imshow('as', img)
    cv2.waitKey(0)

