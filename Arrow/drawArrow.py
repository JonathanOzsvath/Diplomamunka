import cv2
import math
import numpy as np


def calcDistance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def showImageWithClicks(mouseClicks, radius=3, color=(0, 255, 0)):
    imgCopy = img.copy()
    width, height = imgCopy.shape[1], imgCopy.shape[0]
    img_mask = np.zeros((height, width), dtype=np.uint8)

    if len(mouseClicks) < 3:
        for x, y in mouseClicks:
            cv2.circle(imgCopy, (x, y), radius, color, thickness=-1)
    else:
        mouseClicks = np.array(mouseClicks)
        imgCopy = cv2.fillPoly(imgCopy, [mouseClicks], color)
        img_mask = cv2.fillPoly(img_mask, [mouseClicks], 255)

        # arrow.clear()
        # for y in range(0, height):
        #     for x in range(0, width):
        #         if img_mask[y, x] == 255:
        #             arrow.append((x, y))

    cv2.imshow(name, imgCopy)
    cv2.imshow("mask", img_mask)


def setMouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mouseClicks.append([x, y])
        showImageWithClicks(mouseClicks)

    elif event == cv2.EVENT_RBUTTONUP:
        for click in mouseClicks:
            if calcDistance([x, y], click) < 10:
                mouseClicks.remove(click)
                showImageWithClicks(mouseClicks)


if __name__ == '__main__':
    arrow = []
    notArrow = []
    mouseClicks = []

    name = "darts_with_arrow"
    path = '../images/' + name + '.jpg'

    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    cv2.imshow(name, img)

    cv2.setMouseCallback(name, setMouseCallback)

    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()