import cv2
import numpy as np
import math

width = 780
height = 1040
numberOfCirclePointPerSector = 3
rays = [170, 160, 105, 95, 15.9, 7]


def generateDartBoardRefPoints():
    unitCirclePoints = getMetricReferenceCirclePoints()
    r = rays[0]
    points = []
    for i in unitCirclePoints:
        points.append((int(r * i[0]), int(r * i[1])))

    return points


def getMetricReferenceCirclePoints():
    points = []
    for i in np.arange(-81, 279, 360 / 20):
        points.append((np.float32(math.cos(math.radians(i))), np.float32(math.sin(math.radians(i)))))

    return points


def getMetricCirclePoints(numberOfCirclePointPerSector):
    numberOfCirclePoint = 20 * numberOfCirclePointPerSector
    points = []
    for i in np.arange(-81, 279, 360 / numberOfCirclePoint):
        points.append((np.float32(math.cos(math.radians(i))), np.float32(math.sin(math.radians(i)))))

    return points


def generateDartBoardEdgePoints(numberOfCirclePointPerSector):
    refPoints = generateDartBoardRefPoints()
    circlePoints = []

    unitCirclePoints = getMetricCirclePoints(numberOfCirclePointPerSector)

    for r in rays:
        for i in unitCirclePoints:
            circlePoints.append((int(r * i[0]), int(r * i[1])))

    circlePoints.append((0, 0))

    return refPoints, circlePoints


# az első 20 pont a referencia pontok, az utolsó a tábla közepe, a többi pedig 6 * numberOfCirclepoint a külső körívtől befele haladva
def drawDartBoard(img, referencePoints, circlePoints, numberOfCirclePointPerSector, color=(0, 255, 0), shift=(0, 0), savePath=''):
    numberOfCirclePoint = 20 * numberOfCirclePointPerSector
    middlePoint = []
    for i in range(0, numberOfCirclePoint, numberOfCirclePointPerSector):
        middlePoint.append(circlePoints[4 * numberOfCirclePoint + i])

    for i in range(0, 20):
        refPoint = (int(referencePoints[i][0] + shift[0]), int(referencePoints[i][1] + shift[1]))
        midPoint = (int(middlePoint[i][0] + shift[0]), int(middlePoint[i][1] + shift[1]))
        img = cv2.circle(img, refPoint, 3, (0, 0, 255), thickness=-1)
        img = cv2.line(img, refPoint, midPoint, color, thickness=2)

    for i in range(0, 6):
        for j in range(0, numberOfCirclePoint - 1):
            point1 = (int(circlePoints[i * numberOfCirclePoint + j][0] + shift[0]),
                      int(circlePoints[i * numberOfCirclePoint + j][1] + shift[1]))
            point2 = (int(circlePoints[i * numberOfCirclePoint + j + 1][0] + shift[0]),
                      int(circlePoints[i * numberOfCirclePoint + j + 1][1] + shift[1]))
            img = cv2.line(img, point1, point2, color, thickness=2)
        point1 = (
            int(circlePoints[i * numberOfCirclePoint][0] + shift[0]), int(circlePoints[i * numberOfCirclePoint][1] + shift[1]))
        point2 = (int(circlePoints[(i + 1) * numberOfCirclePoint - 1][0] + shift[0]),
                  int(circlePoints[(i + 1) * numberOfCirclePoint - 1][1] + shift[1]))
        img = cv2.line(img, point1, point2, color, thickness=2)

    if savePath != '':
        cv2.imwrite(savePath, img)

    cv2.imshow("img", img)

    cv2.waitKey(0)


def main():
    refPoints, circlePoints = generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    drawDartBoard(img, refPoints, circlePoints, numberOfCirclePointPerSector, shift=(width / 2, height / 2))


if __name__ == '__main__':
    main()
