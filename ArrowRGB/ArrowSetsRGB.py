import os
import cv2
import numpy as np
import global_variable as gv


def saveArrowSets(arrow, notArrow):
    with open('output/arrowSet.txt', 'a') as f:
        for b, g, r in arrow:
            f.write('{}, {}, {}\n'.format(b, g, r))

    with open('output/notArrowSet.txt', 'a') as f:
        for b, g, r in notArrow:
            f.write('{}, {}, {}\n'.format(b, g, r))


def loadArrowSets():
    arrow = []
    notArrow = []

    with open('output/arrowSet.txt', 'r') as f:
        for line in f:
            b, g, r = line.strip().split(',')
            arrow.append((int(b), int(g), int(r)))

    with open('output/notArrowSet.txt', 'r') as f:
        for line in f:
            b, g, r = line.strip().split(',')
            notArrow.append((int(b), int(g), int(r)))

    return arrow, notArrow


def getMaskRGB(name, img, mask):
    maskImg = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    height, width = mask.shape[:2]
    img_points = np.zeros((height, width), dtype=np.uint8)
    img_mask_RGB = np.zeros(img.shape, dtype=np.uint8)
    points = []

    for y in range(0, height):
        for x in range(0, width):
            if maskImg[y, x] > 200:
                points.append(img[y, x])
                img_points[y, x] = 255
                img_mask_RGB[y, x] = img[y, x]
            else:
                img_mask_RGB[y, x] = [0, 0, 0]

    cv2.imwrite("output/" + name_perspective + "/" + name + "_rgb.jpg", img_mask_RGB)
    cv2.imwrite("output/" + name_perspective + "/" + name + ".jpg", img_points)

    return points


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

    for name_perspective in gv.name_perspectives:
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

        name_perspective_mask_negative = name_perspective + '_mask_negative'
        path_perspective_mask_negative = '../images/' + name_perspective_mask_negative + '.jpg'
        img_perspective_mask_negative = cv2.imread(path_perspective_mask_negative)

        cv2.imwrite(directory + name_perspective + '.jpg', img_perspective)
        cv2.imwrite(directory + name_perspective_mask_positive + '.jpg', img_perspective_mask_positive)
        cv2.imwrite(directory + name_perspective_mask_negative + '.jpg', img_perspective_mask_negative)

        arrow = getMaskRGB(name_perspective_mask_positive, img_perspective, img_perspective_mask_positive)
        notArrow = getMaskRGB(name_perspective_mask_negative, img_perspective, img_perspective_mask_negative)

        saveArrowSets(arrow, notArrow)

    cv2.waitKey(0)
