import os
import cv2
import time
import math
import numpy as np
import image_matcher as im
from tabulate import tabulate
import dart_board
import prefilter
import postfilter

mydata = []

numberOfKeypoint = 500
numberOfCirclePointPerSector = 3
max_correct_radius = 5
minHamming_prefilter = 60
maxHamming_postfilter = 75
maxRatio_postfilter = 0.5


def getRunTime(start, end):
    return str(end - start)


def calcDistance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


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


def getMatchesPointWithHomography(kp_ref, matches, homography_matrix):
    points = [kp_ref[match[0].queryIdx].pt for match in matches]
    return Project(points, homography_matrix)


def evaluate(matches, kp_ref, kp_perspective, truth_points, outputFolderName):
    goodMatches = []
    badMatches = []
    points = [kp_perspective[match[0].trainIdx].pt for match in matches]

    for i in range(0, len(points)):
        d = calcDistance(points[i], truth_points[i])
        if d <= max_correct_radius:
            goodMatches.append(i)
        else:
            badMatches.append(i)

    drawInlierOutlierPoints(img_perspective, points, truth_points, goodMatches, badMatches, outputFolderName)

    return len(goodMatches), len(badMatches)


def drawInlierOutlierPoints(img, points, truth_points, inliers, outliers, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in outliers[:20]:
        bad_point = (int(points[i][0]), int(points[i][1]))
        good_point = (int(truth_points[i][0]), int(truth_points[i][1]))
        img = cv2.circle(img, bad_point, 3, (0, 0, 255), thickness=-1)
        img = cv2.circle(img, good_point, 3, (0, 255, 255), thickness=-1)
        img = cv2.line(img, bad_point, good_point, (255, 0, 0), thickness=2)

    for i in inliers:
        img = cv2.circle(img, (int(points[i][0]), int(points[i][1])), 3, (0, 255, 0), thickness=-1)

    cv2.imwrite(directory + '/' + name_perspective + '.png', img)


def drawPoints(img, points, name, color=(0, 255, 0), isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, (int(point[0]), int(point[1])), 3, color, thickness=-1)

    cv2.imshow(name, img)


def addTableRow(path_ref, minHammming, numberOfRefKeyPoint, numberOfFilteredKeyPoints, path_perspective, numberOfPerspectiveKeyPoint,
                detectRunTime, methodName, crossCheck, maxHamming_postfilter, maxRatio_postfilter, foundMatches,
                foundMatchesPercent, numberOfInliers, correctMatchPercent, matchingRunTime):
    mydata.append([path_ref, minHammming, numberOfRefKeyPoint, numberOfFilteredKeyPoints, path_perspective, numberOfPerspectiveKeyPoint,
                  detectRunTime, methodName, crossCheck, maxHamming_postfilter, maxRatio_postfilter, foundMatches,
                  foundMatchesPercent, numberOfInliers, correctMatchPercent, matchingRunTime])


if __name__ == '__main__':
    name_ref = "darts1_2"
    path_ref = '../images/' + name_ref + '.jpg'
    name_perspective = "darts2_2"
    path_perspective = '../images/' + name_perspective + '.jpg'
    outputName = "cvBF_noprefilter_nocheck_nohamming_noratio"
    crossCheck = False

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    click_point_ref = LoadPoints(os.path.splitext(path_ref)[0] + '.click')
    click_point_perspective = LoadPoints(os.path.splitext(path_perspective)[0] + '.click')

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    homography_matrix_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))
    homography_matrix_perspective, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_perspective))
    homography_matrix_ground_truth = np.dot(homography_matrix_perspective, np.linalg.inv(homography_matrix_ref))

    # Initiate ORB
    start = time.time()
    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)
    end = time.time()
    detectRunTime = getRunTime(start, end)

    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    numberOfFilteredKeyPoints = numberOfKeypoint - len(kp_ref)

    # index_ref, index_perspective = \
    # im.BF(kp_ref, des_ref, kp_perspective, des_perspective)

    start = time.time()
    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=crossCheck)
    end = time.time()
    matchingRunTime = getRunTime(start, end)

    matches = postfilter.distanceFilter(matches, maxHamming=maxHamming_postfilter)
    matches = postfilter.ratioFilter(matches, maxRatio=maxHamming_postfilter)

    truth_points = getMatchesPointWithHomography(kp_ref, matches, homography_matrix_ground_truth)
    numberOfInliers, numberOfOutliers = evaluate(matches, kp_ref, kp_perspective, truth_points, outputName)

    foundMatches = len(matches)

    # mydata = [(
    #     path_ref, "-", str(numberOfKeypoint), "-", path_perspective, str(numberOfKeypoint), detectRunTime,
    #     outputName, crossCheck, "-", "-", foundMatches, str(foundMatches / numberOfKeypoint * 100),
    #     numberOfInliers, str(numberOfInliers / foundMatches * 100), matchingRunTime)]

    addTableRow(path_ref, minHamming_prefilter, numberOfKeypoint, numberOfFilteredKeyPoints, path_perspective,
                numberOfKeypoint, detectRunTime, outputName, crossCheck, maxHamming_postfilter, maxRatio_postfilter,
                foundMatches, str(foundMatches / numberOfKeypoint * 100), numberOfInliers,
                str(numberOfInliers / foundMatches * 100), matchingRunTime)

    headers = ["Referenciakép neve", "Előszűrés küszöbértéke", "Detektált pontok száma", "Szűrt pontok száma",
               "Perspektív kép neve", "Detektált pontok száma perspektív képen", "Detektálás futási ideje (ms)",
               "Párkeresési módszer", "Cross check", "Távolság utószűrés paramétere", "Arány utószűrés paramétere",
               "Megtalált párok száma", "Párosított pontok %-a", "Helyes párok száma", "Helyes párok %-a",
               "Párkeresés futási ideje (ms)"]

    print(tabulate(mydata, headers=headers))

    cv2.waitKey(0)
