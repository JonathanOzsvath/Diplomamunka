import cv2
import numpy as np
import math

width = 780
height = 1040


def getMetricCirclePoints(n):
    points = []
    for i in np.arange(-81, 279, 360/n):
        points.append((math.cos(math.radians(i)), math.sin(math.radians(i))))

    return points


def generateDartBoardEdgePoints():
    unitCirclePoints = getMetricCirclePoints(20)
    rays = [170, 162, 107, 99, 15.9, 6.35]
    rays = [i*2 for i in rays]
    points = []

    for r in rays:
        for i in unitCirclePoints:
            points.append((int(r * i[0]), int(r * i[1])))

    return points


def drawDartBoard(img, points, color=(0, 255, 0)):

    for point in points:
        point = (int(point[0] + width/2), int(point[1] + height/2))
        img = cv2.circle(img, point, 2, color, thickness=-1)

    for i in range(0, 20):
        point = (int(points[i][0] + width / 2), int(points[i][1] + height / 2))
        img = cv2.line(img, point, (int(width / 2), int(height / 2)), (255, 0, 0))

    cv2.imshow("img", img)

    cv2.waitKey(0)


def main():
    points = generateDartBoardEdgePoints()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    drawDartBoard(img, points)


if __name__ == '__main__':
    main()
