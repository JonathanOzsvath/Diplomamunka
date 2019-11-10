import os
import cv2
import math
import numpy as np
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import RANSAC
import dart_board
import cutOutside as co


def saveArrowSets(arrow, notArrow):
    with open('output/arrowSet.txt', 'a') as f:
        for x, y in arrow:
            f.write('{}, {}\n'.format(x, y))

    with open('output/notArrowSet.txt', 'a') as f:
        for x, y in notArrow:
            f.write('{}, {}\n'.format(x, y))


def loadArrowSets():
    arrow = []
    notArrow = []

    with open('output/arrowSet.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            arrow.append((int(x), int(y)))

    with open('output/notArrowSet.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            notArrow.append((int(x), int(y)))

    return arrow, notArrow


def getArrow_notArrow(img_YUV, maskImg_positive, maskImg_negative):
    maskImg_positive = cv2.cvtColor(maskImg_positive, cv2.COLOR_BGR2GRAY)
    maskImg_negative = cv2.cvtColor(maskImg_negative, cv2.COLOR_BGR2GRAY)

    height, width = maskImg_positive.shape[:2]
    img_arrow = np.zeros((height, width), dtype=np.uint8)
    img_notArrow = np.zeros((height, width), dtype=np.uint8)
    arrow = []
    notArrow = []

    for y in range(0, height):
        for x in range(0, width):
            if maskImg_positive[y, x] == 255 and maskImg_negative[y, x] == 255:
                arrow.append(img_YUV[y, x][1:])
                img_arrow[y, x] = 255
            elif maskImg_positive[y, x] == 0 and maskImg_negative[y, x] == 255:
                notArrow.append(img_YUV[y, x][1:])
                img_notArrow[y, x] = 255

    # cv2.imwrite("output/arrow.jpg", img_arrow)
    # cv2.imwrite("output/notArrow.jpg", img_notArrow)

    return arrow, notArrow


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filePath = "output/arrowSet.txt"
    if os.path.exists(filePath):
        os.remove(filePath)

    filePath = "output/notArrowSet.txt"
    if os.path.exists(filePath):
        os.remove(filePath)

    name_perspectives = ['darts_with_arrow', 'darts_with_arrow2', 'darts_with_arrow3', 'darts_with_arrow4', 'darts_with_arrow5', 'darts_with_arrow6',
                         'darts_with_arrow7', 'darts_with_arrow8', 'darts_with_arrow9', 'darts_with_arrow10']
    # name_perspectives = ['darts_with_arrow']

    for name_perspective in name_perspectives:
        directory = "output/" + name_perspective + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        path_perspective = '../images/' + name_perspective + '.jpg'
        img_perspective = cv2.imread(path_perspective)
        img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        img_YUV = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2YUV)

        name_perspective_mask_positive = name_perspective + '_mask_positive'
        path_perspective_mask_positive = '../images/' + name_perspective_mask_positive + '.jpg'
        img_perspective_mask_positive = cv2.imread(path_perspective_mask_positive)
        img_perspective_mask_positive = cv2.resize(img_perspective_mask_positive, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        name_perspective_mask_negative = name_perspective + '_mask_negative'
        path_perspective_mask_negative = '../images/' + name_perspective_mask_negative + '.jpg'
        img_perspective_mask_negative = cv2.imread(path_perspective_mask_negative)
        img_perspective_mask_negative = cv2.resize(img_perspective_mask_negative, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(directory + name_perspective + '.jpg', img_perspective)
        cv2.imwrite(directory + name_perspective_mask_positive + '.jpg', img_perspective_mask_positive)
        cv2.imwrite(directory + name_perspective_mask_negative + '.jpg', img_perspective_mask_negative)

        arrow, notArrow = getArrow_notArrow(img_YUV, img_perspective_mask_positive, img_perspective_mask_negative)

        saveArrowSets(arrow, notArrow)

    cv2.waitKey(0)
