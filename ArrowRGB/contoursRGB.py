import os
import cv2
import math
import numpy as np

if __name__ == '__main__':
    name = 'darts_with_arrow12_probability_0.3'
    path = '../images/' + name + '.jpg'
    img = cv2.imread(path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', img1)

    # img1 = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=11)
    # img1 = cv2.GaussianBlur(img1, (3, 3), 2)
    # img1 = cv2.Laplacian(img1, cv2.CV_8U)
    # img1 = cv2.medianBlur(img1, 11)
    # img1 = cv2.Canny(img1, 30, 200)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # dilated = cv2.dilate(img1, kernel)
    # cv2.imshow('blur', img1)

    img2 = img1.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     cv2.drawContours(img2, [c], 0, (0, 0, 255), 3)
    #     cv2.imshow("contours", img2)
    #     cv2.waitKey(0)

    for c in contours:
        area = cv2.contourArea(c)

        if  1000 < area < 1500:
            hull = cv2.convexHull(c)
            cv2.drawContours(img2, [hull], 0, (0, 0, 255), 1)
            cv2.imshow("contours", img2)

    cv2.waitKey(0)
