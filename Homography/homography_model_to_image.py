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


def Project(points, homograpy_matrix):


    # TODO
    return None


if __name__ == '__main__':
    # referencia kép
    img = cv2.imread('../images/' + imageName + '.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow(imageName, img)

    click_points = LoadPoints(imageName + '.click')
    metric_points = dart_board.getMetricCirclePoints(20)

    p = np.array([1.0, 2.0])
    homography_matrix, _ = cv2.findHomography(np.array(metric_points), np.array(click_points))
    proj_points = Project(metric_points, homography_matrix)

    cv2.waitKey(0)
