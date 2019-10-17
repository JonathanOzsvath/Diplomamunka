import cv2
import math
import sys
import os

mouseClicks = []
imagePath = '../images/darts1_1.jpg'
deleteSquareSide = 3


def showImageWithClicks(radius=3, color=(0, 255, 0)):
    imgCopy = img.copy()
    for click in mouseClicks:
        cv2.circle(imgCopy, click, radius, color, thickness=-1)

    cv2.imshow(imageName, imgCopy)


def setMouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mouseClicks.append((x, y))
        showImageWithClicks()

    elif event == cv2.EVENT_RBUTTONUP:
        for click in mouseClicks:
            if calcDistance((x, y), click) < 10:
                mouseClicks.remove(click)
                showImageWithClicks()


def calcDistance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def savePoints():
    with open(imageName + '.click', 'w') as f:
        for x, y in mouseClicks:
            f.write('{}, {}\n'.format(x, y))


if __name__ == '__main__':

    argv = sys.argv

    if len(sys.argv) == 2:
        imagePath = sys.argv[1]

    imageName = os.path.splitext(imagePath)[0]

    # referencia kép
    img = cv2.imread(imagePath)

    if img is None:
        print("Nem sikerült a képet beolvasni!!!")
        exit(1)

    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow(imageName, img)

    cv2.setMouseCallback(imageName, setMouseCallback)
    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()
    elif k == 13:
        savePoints()
