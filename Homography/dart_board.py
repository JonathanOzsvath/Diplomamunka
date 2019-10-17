import cv2
import numpy as np
import math

width = 780
height = 1040
numberOfCirclepoint = 20
rays = [170, 162, 107, 99, 15.9, 6.35]


def generateDartBoardRefPoints():
    unitCirclePoints = getMetricCirclePoints()
    r = rays[0]
    points = []
    for i in unitCirclePoints:
        points.append((int(r * i[0]), int(r * i[1])))

    return points


def getMetricCirclePoints():
    points = []
    for i in np.arange(-81, 279, 360 / numberOfCirclepoint):
        points.append((np.float32(math.cos(math.radians(i))), np.float32(math.sin(math.radians(i)))))

    return points


def generateDartBoardEdgePoints():
    unitCirclePoints = getMetricCirclePoints()
    points = []

    for r in rays:
        for i in unitCirclePoints:
            points.append((int(r * i[0]), int(r * i[1])))

    points.append((0.0, 0.0))

    return points


def drawDartBoard(img, points, color=(0, 255, 0), shift=(0, 0), savePath=''):
    for point in points:
        point = (int(point[0] + shift[0]), int(point[1] + shift[1]))
        img = cv2.circle(img, point, 2, color, thickness=-1)

    for i in range(0, 6):
        for j in range(0, 19):
            point1 = (int(points[i * numberOfCirclepoint + j][0] + shift[0]),
                      int(points[i * numberOfCirclepoint + j][1] + shift[1]))
            point2 = (int(points[i * numberOfCirclepoint + j + 1][0] + shift[0]),
                      int(points[i * numberOfCirclepoint + j + 1][1] + shift[1]))
            img = cv2.line(img, point1, point2, color)
        point1 = (
            int(points[i * numberOfCirclepoint][0] + shift[0]), int(points[i * numberOfCirclepoint][1] + shift[1]))
        point2 = (int(points[(i + 1) * numberOfCirclepoint - 1][0] + shift[0]),
                  int(points[(i + 1) * numberOfCirclepoint - 1][1] + shift[1]))
        img = cv2.line(img, point1, point2, color)

    for j in range(0, 20):
        point = (int(points[j][0] + shift[0]), int(points[j][1] + shift[1]))
        middlePoint = points[-1]
        img = cv2.line(img, point, (int(middlePoint[0] + shift[0]), int(middlePoint[1] + shift[1])), color)

    if savePath != '':
        cv2.imwrite(savePath, img)

    cv2.imshow("img", img)

    cv2.waitKey(0)


def main():
    points = generateDartBoardEdgePoints()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    drawDartBoard(img, points, shift=(width / 2, height / 2))


if __name__ == '__main__':
    main()
