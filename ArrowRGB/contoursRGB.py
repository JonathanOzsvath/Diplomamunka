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
    img1 = cv2.medianBlur(img1, 11)
    img1 = cv2.Canny(img1, 30, 200)
    cv2.imshow('blur', img1)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # find minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
        #
        # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        # cv2.imshow("contours", img)

        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(c)
        cv2.drawContours(img, hull, 0, (0, 0, 255), 3)
        cv2.imshow("contours", img)

    cv2.waitKey(0)
