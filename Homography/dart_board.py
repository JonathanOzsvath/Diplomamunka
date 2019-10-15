import cv2
import numpy as np
import math

width = 780
height = 1040
numberOfCirclepoint = 20
rays = [170, 162, 107, 99, 15.9, 6.35]


# rays = [i * 2 for i in rays]


def generateDartBoardRefPoints():
    unitCirclePoints = getMetricCirclePoints(numberOfCirclepoint)
    r = rays[0]
    points = []
    for i in unitCirclePoints:
        points.append((int(r * i[0] + width / 2), int(r * i[1] + height / 2)))

    return points


def getMetricCirclePoints(n):
    points = []
    for i in np.arange(-81, 279, 360 / n):
        points.append((np.float32(math.cos(math.radians(i))), np.float32(math.sin(math.radians(i)))))

    return points


def generateDartBoardEdgePoints():
    unitCirclePoints = getMetricCirclePoints(numberOfCirclepoint)
    points = []

    for r in rays:
        for i in unitCirclePoints:
            points.append((int(r * i[0]), int(r * i[1])))

    return points


def drawDartBoard(img, points, color=(0, 255, 0)):
    for point in points:
        point = (int(point[0] + width / 2), int(point[1] + height / 2))
        img = cv2.circle(img, point, 2, color, thickness=-1)

    for i in range(0, 6):
        for j in range(0, 19):
            point1 = (int(points[i * numberOfCirclepoint + j][0] + width / 2),
                      int(points[i * numberOfCirclepoint + j][1] + height / 2))
            point2 = (int(points[i * numberOfCirclepoint + j + 1][0] + width / 2),
                      int(points[i * numberOfCirclepoint + j + 1][1] + height / 2))
            img = cv2.line(img, point1, point2, color)
        point1 = (
            int(points[i * numberOfCirclepoint][0] + width / 2), int(points[i * numberOfCirclepoint][1] + height / 2))
        point2 = (int(points[(i + 1) * numberOfCirclepoint - 1][0] + width / 2),
                  int(points[(i + 1) * numberOfCirclepoint - 1][1] + height / 2))
        img = cv2.line(img, point1, point2, color)

    for j in range(0, 20):
        point = (int(points[j][0] + width / 2), int(points[j][1] + height / 2))
        img = cv2.line(img, point, (int(width / 2), int(height / 2)), color)
        img = cv2.circle(img, point, 3, (0, 0, 255), thickness=-1)

    cv2.imshow("img", img)

    cv2.waitKey(0)


def main():
    points = generateDartBoardEdgePoints()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    drawDartBoard(img, points)


if __name__ == '__main__':
    main()
