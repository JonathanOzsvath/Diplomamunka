import cv2
import numpy as np
import Homography.dart_board as dart_board

imageName = 'darts1_2'


def LoadPoints(filename):
    mouseClicks = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            mouseClicks.append((np.float32(x), np.float32(y)))

    return mouseClicks


def Project(points, homography_matrix):
    newPoints = []

    for x, y in points:
        point = np.array([[x], [y], [1]])
        newPoint = np.dot(homography_matrix, point)
        newPoint = (newPoint[0]/newPoint[2], newPoint[1]/newPoint[2])
        newPoints.append(newPoint)

    return newPoints


if __name__ == '__main__':
    # referencia kép
    img = cv2.imread('../images/' + imageName + '.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow(imageName, img)

    click_points = LoadPoints(imageName + '.click')
    metric_points = dart_board.generateDartBoardRefPoints()

    p = np.array([1.0, 2.0])
    homography_matrix, _ = cv2.findHomography(np.array(metric_points), np.array(click_points))
    proj_points = Project(metric_points, homography_matrix)

    for point in proj_points:
        img = cv2.circle(img, point, 3, (0, 255, 255), thickness=-1)

    cv2.imshow("homography", img)

    cv2.waitKey(0)
