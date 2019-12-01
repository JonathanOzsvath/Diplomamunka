import cv2
import numpy as np
import math
import os

width = 780
height = 1040
numberOfCirclePointPerSector = 10
rays = [170, 160, 105, 95, 15.9, 7]


def generateDartBoardRefPoints():
    unitCirclePoints = getMetricReferenceCirclePoints()
    r = rays[0]
    points = []
    for i in unitCirclePoints:
        points.append((int((round(r * i[0]))), int(round(r * i[1]))))

    return points


def getMetricReferenceCirclePoints():
    points = []
    for i in np.arange(-81, 279, 360 / 20):
        points.append((np.float64(math.cos(math.radians(i))), np.float64(math.sin(math.radians(i)))))

    return points


def getMetricCirclePoints(numberOfCirclePointPerSector):
    numberOfCirclePoint = 20 * numberOfCirclePointPerSector
    points = []
    for i in np.arange(-81, 279, 360 / numberOfCirclePoint):
        points.append((np.float64(math.cos(math.radians(i))), np.float64(math.sin(math.radians(i)))))

    return points


def generateDartBoardEdgePoints(numberOfCirclePointPerSector):
    refPoints = generateDartBoardRefPoints()
    circlePoints = []

    unitCirclePoints = getMetricCirclePoints(numberOfCirclePointPerSector)

    for r in rays:
        for i in unitCirclePoints:
            circlePoints.append((round(r * i[0]), round(r * i[1])))

    circlePoints.append((0, 0))

    return refPoints, circlePoints


def drawDartBoard(img, referencePoints, circlePoints, numberOfCirclePointPerSector, color=(0, 255, 0), shift=(0, 0), savePath='', showImage=True):
    numberOfCirclePoint = 20 * numberOfCirclePointPerSector
    middlePoint = []
    for i in range(0, numberOfCirclePoint, numberOfCirclePointPerSector):
        middlePoint.append(circlePoints[4 * numberOfCirclePoint + i])



    for i in range(0, 6):
        for j in range(0, numberOfCirclePoint - 1):
            point1 = (int(round(circlePoints[i * numberOfCirclePoint + j][0] + shift[0])),
                      int(round(circlePoints[i * numberOfCirclePoint + j][1] + shift[1])))
            point2 = (int(round(circlePoints[i * numberOfCirclePoint + j + 1][0] + shift[0])),
                      int(round(circlePoints[i * numberOfCirclePoint + j + 1][1] + shift[1])))
            img = cv2.line(img, point1, point2, color, thickness=2, lineType=cv2.LINE_AA)
        point1 = (
            int(round(circlePoints[i * numberOfCirclePoint][0] + shift[0])), int(round(circlePoints[i * numberOfCirclePoint][1] + shift[1])))
        point2 = (int(round(circlePoints[(i + 1) * numberOfCirclePoint - 1][0] + shift[0])),
                  int(round(circlePoints[(i + 1) * numberOfCirclePoint - 1][1] + shift[1])))
        img = cv2.line(img, point1, point2, color, thickness=2, lineType=cv2.LINE_AA)

    for i in range(0, 20):
        refPoint = (int(round(referencePoints[i][0] + shift[0])), int(round(referencePoints[i][1] + shift[1])))
        midPoint = (int(round(middlePoint[i][0] + shift[0])), int(round(middlePoint[i][1] + shift[1])))
        img = cv2.line(img, refPoint, midPoint, color, thickness=2, lineType=cv2.LINE_AA)
        img = cv2.circle(img, refPoint, 3, (0, 0, 255), thickness=-1)

    if savePath != '':
        cv2.imwrite(savePath, img)

    if showImage:
        cv2.imshow("img", img)

    cv2.waitKey(0)
    return img


def main():
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    refPoints, circlePoints = generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = 255
    drawDartBoard(img, refPoints, circlePoints, numberOfCirclePointPerSector, shift=(width / 2, height / 2), savePath='output/ref.png')


if __name__ == '__main__':
    main()
