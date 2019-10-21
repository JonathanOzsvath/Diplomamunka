import cv2
import sys
import os
import numpy as np
import math
import statistics
import dart_board

imagePath = '../images/darts1_1.jpg'
numberOfCirclePointPerSector = 10


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
        newPoint = (int(round((newPoint[0] / newPoint[2])[0])), int(round((newPoint[1] / newPoint[2])[0])))
        newPoints.append(newPoint)

    return newPoints


def calcDistance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def ComputeResidualErrors(proj_points, click_points):
    error_vector = []

    for i in range(len(proj_points)):
        error_vector.append(calcDistance(proj_points[i], click_points[i]))

    return error_vector


def drawReferenceToHomographyPoints(img, click_points, proj_points):
    for i in range(0, len(proj_points)):
        img = cv2.circle(img, click_points[i], 3, (0, 0, 255), thickness=-1)
        img = cv2.circle(img, proj_points[i], 3, (0, 255, 0), thickness=-1)
        img = cv2.line(img, proj_points[i], click_points[i], (0, 255, 255), thickness=2)

    cv2.imshow("homography", img)
    cv2.imwrite("output/ReferenceToHomographyPoints.jpg", img)


def drawPoints(img, points, color):
    for point in points:
        img = cv2.circle(img, point, 3, color, thickness=-1)

    cv2.imshow("Point", img)


if __name__ == '__main__':

    argv = sys.argv

    if len(sys.argv) == 2:
        imagePath = sys.argv[1]

    imageName = os.path.splitext(imagePath)[0]
    name = imageName.split('/')[-1]

    # referencia kép
    img = cv2.imread(imagePath)

    if img is None:
        print("Nem sikerült a képet beolvasni!!!")
        exit(1)

    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # cv2.imshow(imageName, img)
    click_points = LoadPoints(imageName + '.click')
    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    homography_matrix, _ = cv2.findHomography(np.array(refPoints), np.array(click_points))
    proj_refPoints = Project(refPoints, homography_matrix)

    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # drawReferenceToHomographyPoints(img, click_points, proj_points)

    error_vector = ComputeResidualErrors(proj_refPoints, click_points)

    eredmenyPath = "output/eredmenyek.txt"
    with open("output/eredmenyek.txt", 'a') as f1, open(eredmenyPath, 'r') as f2:
        contains = False

        for line in f2:
            if line.find(imagePath) != -1:
                contains = True

        if not contains:
            f1.write('imagePath: ' + imagePath + ':\n')
            f1.write("\tAvg: " + str(statistics.mean(error_vector)) + '\n')
            f1.write("\tMax: " + str(max(error_vector)) + '\n')

    print("Error vektor átlaga: " + str(statistics.mean(error_vector)))
    print("Error vektor maximuma: " + str(max(error_vector)))

    proj_circlepoints = Project(circlePoints, homography_matrix)

    dart_board.drawDartBoard(img, proj_refPoints, proj_circlepoints, numberOfCirclePointPerSector, (0, 255, 0), savePath="output/" + name + "_homography_model_to_image.png")

    cv2.waitKey(0)
