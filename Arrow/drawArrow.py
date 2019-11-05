import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

isLButtonDown = False
isRButtonDown = False
isArrow = True


def calcDistance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def showImageWithClicks(radius=3, color=(0, 255, 0)):
    imgCopy = img.copy()
    for point in arrow:
        cv2.circle(imgCopy, point, radius, color, thickness=-1)

    for point in notArrow:
        cv2.circle(imgCopy, point, radius, (0, 0, 255), thickness=-1)

    cv2.imshow(name, imgCopy)


def setMouseCallback(event, x, y, flags, param):
    global isLButtonDown, isRButtonDown, isArrow

    if event == cv2.EVENT_LBUTTONDOWN:
        isLButtonDown = True

    if event == cv2.EVENT_LBUTTONUP:
        isLButtonDown = False

    if isLButtonDown:
        if isArrow:
            arrow.add((x, y))
            showImageWithClicks(radius=1)
        else:
            notArrow.add((x,y))
            showImageWithClicks(radius=1)

    if event == cv2.EVENT_RBUTTONDOWN:
        isRButtonDown = True

    if event == cv2.EVENT_RBUTTONUP:
        isRButtonDown = False

    if isRButtonDown:
        if isArrow:
            for point in arrow.copy():
                if calcDistance((x, y), point) < 5:
                    arrow.remove(point)
                    showImageWithClicks(radius=1)
        else:
            for point in notArrow.copy():
                if calcDistance((x, y), point) < 5:
                    notArrow.remove(point)
                    showImageWithClicks(radius=1)


if __name__ == '__main__':
    arrow = set()
    notArrow = set()
    mouseClicks = []

    name = "darts_with_arrow"
    path = '../images/' + name + '.jpg'

    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    cv2.imshow(name, img)

    cv2.setMouseCallback(name, setMouseCallback)
    while cv2.getWindowProperty(name, 0) >= 0:

        k = cv2.waitKey(0)

        if k == 27:
            cv2.destroyAllWindows()
        elif k == 13:
            isArrow = not isArrow
            print(isArrow)

    u_arrow = [img_yuv[y, x][1] for x, y in arrow]
    v_arrow = [img_yuv[y, x][2] for x, y in arrow]

    u_notArrow = [img_yuv[y, x][1] for x, y in notArrow]
    v_notArrow = [img_yuv[y, x][2] for x, y in notArrow]

    # plt.plot(u_arrow, v_arrow, 'gx')
    # plt.plot(u_notArrow, v_notArrow, 'rx')
    # plt.xlabel('U')
    # plt.ylabel('V')

    plt.plot(v_arrow, u_arrow, 'gx')
    plt.plot(v_notArrow, u_notArrow, 'rx')
    plt.xlabel('V')
    plt.ylabel('U')

# -(x - 150)² + 125 - y

    plt.show()
