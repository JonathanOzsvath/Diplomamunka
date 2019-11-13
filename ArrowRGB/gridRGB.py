import os
import cv2
import math
import statistics
import numpy as np
import ArrowSetsRGB
import global_variable as gv


def computeColorHist3D(arrows, notArrows, bmin=0, gmin=0, rmin=0, db=10, dg=10, dr=10):
    color_histogram_positive = np.zeros((math.ceil(255 / db), math.ceil(255 / dg), math.ceil(255 / dr)), dtype=np.int)
    color_histogram_negative = np.zeros((math.ceil(255 / db), math.ceil(255 / dg), math.ceil(255 / dr)), dtype=np.int)

    for b, g, r in arrows:
        i = math.floor((b - bmin) / db)
        j = math.floor((g - gmin) / dg)
        k = math.floor((r - rmin) / dr)
        color_histogram_positive[i, j, k] = color_histogram_positive[i, j, k] + 1

    for b, g, r in notArrows:
        i = math.floor((b - bmin) / db)
        j = math.floor((g - gmin) / dg)
        k = math.floor((r - rmin) / dr)
        color_histogram_negative[i, j, k] = color_histogram_negative[i, j, k] + 1

    print('Arrows: {}'.format(len(arrows)))
    print('Not Arrows: {}'.format(len(notArrows)))
    return color_histogram_positive, color_histogram_negative


def computePMatrix(color_histogram_positive, color_histogram_negative):
    width, height, depth = color_histogram_positive.shape
    P_positive = np.zeros(color_histogram_positive.shape)

    sum = np.add(color_histogram_positive, color_histogram_negative)

    for i in range(width):
        for j in range(height):
            for k in range(depth):
                if sum[i, j, k] != 0:
                    P_positive[i, j, k] = color_histogram_positive[i, j, k] / sum[i, j, k]
                else:
                    P_positive[i, j, k] = 0

    return P_positive


def probabilitySegmentation(name, img, P_positive, bmin=0, gmin=0, rmin=0, db=10, dg=10, dr=10):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img.shape[:2]
    img_P = np.zeros(img.shape[:2], dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            i = math.floor((b - bmin) / db)
            j = math.floor((g - gmin) / dg)
            k = math.floor((r - rmin) / dr)

            img_P[y, x] = int(P_positive[i, j, k] * 255)

    img_P = cv2.applyColorMap(img_P, cv2.COLORMAP_JET)

    # cv2.imshow('probabilitySegmentation', img_P)
    cv2.imwrite(directory + name + '_probability.jpg', img_P)


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    arrows, notArrows = ArrowSetsRGB.loadArrowSets()

    color_histogram_positive, color_histogram_negative = computeColorHist3D(arrows, notArrows, 0, 0, 0, gv.d, gv.d, gv.d)

    P_positive = computePMatrix(color_histogram_positive, color_histogram_negative)

    for name in gv.name_perspectives:
        path = '../images/' + name + '.jpg'
        img = cv2.imread(path)
        img_resize = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        probabilitySegmentation(name, img_resize, P_positive, 0, 0, 0, gv.d, gv.d, gv.d)

    cv2.waitKey(0)
