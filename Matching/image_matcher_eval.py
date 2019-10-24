import os
import cv2
import time
import math
import statistics
import numpy as np
import image_matcher as im
from tabulate import tabulate
import dart_board
import prefilter
import postfilter

mydata = []
summary = []

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
max_correct_radius = 5


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


def evaluate(img, matches, kp_perspective, truth_points, outputFolderName):
    goodMatches = []
    badMatches = []
    points = [kp_perspective[match[0].trainIdx].pt for match in matches]

    for i in range(0, len(points)):
        d = calcDistance(points[i], truth_points[i])
        if d <= max_correct_radius:
            goodMatches.append(i)
        else:
            badMatches.append(i)

    drawInlierOutlierPoints(img, kp_perspective, truth_points, goodMatches, badMatches, outputFolderName)

    return len(goodMatches), len(badMatches)


def drawInlierOutlierPoints(img, kp, truth_points, inliers, outliers, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in outliers:
        bad_point = (int(kp[i].pt[0]), int(kp[i].pt[1]))
        good_point = (int(truth_points[i][0]), int(truth_points[i][1]))
        img = cv2.circle(img, bad_point, 3, (0, 0, 255), thickness=-1)
        img = cv2.circle(img, good_point, 3, (0, 255, 255), thickness=-1)
        # img = cv2.line(img, bad_point, good_point, (255, 0, 0), thickness=2)

    for i in inliers:
        img = cv2.circle(img, (int(kp[i].pt[0]), int(kp[i].pt[1])), int(kp[i].size / 2), (0, 255, 0), thickness=1)

    cv2.imwrite(directory + '/' + name_perspective + '.jpg', img)


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


def runMethod(method_name, name_ref, name_perspective, prefilterValue, crossCheck, postFilterHamming, postFilterRatio, outputName):
    path_ref = '../images/' + name_ref + '.jpg'
    path_perspective = '../images/' + name_perspective + '.jpg'

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

    if prefilterValue:
        minHamming_prefilter = prefilterValue
        kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
        numberOfFilteredKeyPoints = numberOfKeypoint - len(kp_ref)
    else:
        minHamming_prefilter = '-'
        numberOfFilteredKeyPoints = '-'

    start = time.time()
    if method_name == 'cvBF':
        matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=crossCheck)
    elif method_name == 'BF':
        matches = im.BF(kp_ref, des_ref, kp_perspective, des_perspective)
    elif method_name == 'FLANN':
        matches = im.flann(des_ref, des_perspective)
    end = time.time()
    matchingRunTime = getRunTime(start, end)

    if postFilterHamming:
        maxHamming_postfilter = postFilterHamming
        matches = postfilter.distanceFilter(matches, maxHamming=maxHamming_postfilter)
    else:
        maxHamming_postfilter = '-'

    if postFilterRatio:
        maxRatio_postfilter = postFilterRatio
        matches = postfilter.ratioFilter(matches, maxRatio=maxRatio_postfilter)
    else:
        maxRatio_postfilter = '-'

    truth_points = getMatchesPointWithHomography(kp_ref, matches, homography_matrix_ground_truth)
    numberOfInliers, numberOfOutliers = evaluate(img_perspective, matches, kp_perspective, truth_points, outputName)

    foundMatches = len(matches)

    addTableRow(path_ref, minHamming_prefilter, numberOfKeypoint, numberOfFilteredKeyPoints, path_perspective,
                numberOfKeypoint, detectRunTime, outputName, crossCheck, maxHamming_postfilter, maxRatio_postfilter,
                foundMatches, (foundMatches / numberOfKeypoint * 100), numberOfInliers,
                (numberOfInliers / foundMatches * 100), matchingRunTime)

    summary.append([path_ref, minHamming_prefilter, numberOfKeypoint, numberOfFilteredKeyPoints, path_perspective,
                    numberOfKeypoint, detectRunTime, outputName, crossCheck, maxHamming_postfilter, maxRatio_postfilter,
                    foundMatches, (foundMatches / numberOfKeypoint * 100), numberOfInliers,
                    (numberOfInliers / foundMatches * 100), matchingRunTime])


def makeMethodName(nameMethod, prefilterValue, crossCheck, postFilterHamming, postFilterRatio):
    s = [nameMethod]
    if prefilterValue:
        s.append(str(prefilterValue))
    else:
        s.append("noprefilter")

    if crossCheck:
        s.append("check")
    else:
        s.append("nocheck")

    if postFilterHamming:
        s.append(str(postFilterHamming))
    else:
        s.append("nohamming")

    if postFilterRatio:
        s.append(str(postFilterRatio))
    else:
        s.append("noratio")

    return '_'.join(s)


def addSummaryRow(minHamming_prefilter):
    if minHamming_prefilter:
        meanFilteredKeyPoints = statistics.mean([float(i[3]) for i in summary])
    else:
        meanFilteredKeyPoints = '-'

    addTableRow(summary[0][0], summary[0][1], summary[0][2], meanFilteredKeyPoints, "Összesítés",
                summary[0][5], statistics.mean([float(i[6]) for i in summary]), summary[0][7], summary[0][8], summary[0][9], summary[0][10],
                statistics.mean([int(i[11]) for i in summary]), statistics.mean([float(i[12]) for i in summary]), statistics.mean([float(i[13]) for i in summary]),
                statistics.mean([float(i[14]) for i in summary]), statistics.mean([float(i[15]) for i in summary]))


if __name__ == '__main__':
    name_ref = "darts1_1"

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']
    # minHamming_prefilters = [False, 50, 60, 70]
    minHamming_prefilters = [False, 70]
    # maxHamming_postfilters = [False, 48, 53, 58]
    maxHamming_postfilters = [False, 53]
    cross_Checks = [True, False]
    # maxRatio_postfilters = [False, 0.6, 0.7, 0.8]
    maxRatio_postfilters = [False, 0.7]
    methodNames = ['cvBF', 'FLANN']

    # name_perspective = "darts2_1"
    # minHamming_prefilter = 60
    # maxHamming_postfilter = 48
    # cross_Check = False
    # maxRatio_postfilter = 0.7
    # methodName = "cvBF"

    for name_perspective in name_perspectives:
        runMethod('BF', name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
                  outputName=makeMethodName('BF', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
        print(makeMethodName('BF', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False) + '_' + name_perspective)
        if len(summary) == len(name_perspectives):
            addSummaryRow(False)
            summary.clear()

    for methodName in methodNames:
        for maxRatio_postfilter in maxRatio_postfilters:
            for cross_Check in cross_Checks:
                if (cross_Check and maxRatio_postfilter) or (methodName == 'FLANN' and cross_Check):
                    continue
                for maxHamming_postfilter in maxHamming_postfilters:
                    for minHamming_prefilter in minHamming_prefilters:
                        for name_perspective in name_perspectives:
                            runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter,
                                      outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter))
                            print(makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter) + '_' + name_perspective)

                            if len(summary) == len(name_perspectives):
                                addSummaryRow(minHamming_prefilter)
                                summary.clear()

    # # 1
    # runMethod('BF', name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName('BF', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
    #
    # runMethod('BF', name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName('BF', prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
    # # 2
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
    #
    # # 3
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=True, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=True, postFilterHamming=False, postFilterRatio=False))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=True, postFilterHamming=False, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=True, postFilterHamming=False, postFilterRatio=False))
    #
    # # 4
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=False))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=False))
    # # 5
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=True, postFilterHamming=maxHamming_postfilter, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=True, postFilterHamming=maxHamming_postfilter, postFilterRatio=False))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=True, postFilterHamming=maxHamming_postfilter, postFilterRatio=False,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=True, postFilterHamming=maxHamming_postfilter, postFilterRatio=False))
    # # 6
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter))
    # # 7
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName(methodName, prefilterValue=False, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter))
    #
    # runMethod(methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter))
    # # FLANN
    # runMethod('FLANN', name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName('FLANN', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter))
    #
    # runMethod('FLANN', name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter,
    #           outputName=makeMethodName('FLANN', prefilterValue=minHamming_prefilter, crossCheck=False, postFilterHamming=False, postFilterRatio=maxRatio_postfilter))

    headers = ["Referenciakép neve", "Előszűrés küszöbértéke", "Detektált pontok száma", "Szűrt pontok száma",
               "Perspektív kép neve", "Detektált pontok száma perspektív képen", "Detektálás futási ideje (ms)",
               "Párkeresési módszer", "Cross check", "Távolság utószűrés paramétere", "Arány utószűrés paramétere",
               "Megtalált párok száma", "Párosított pontok %-a", "Helyes párok száma", "Helyes párok %-a",
               "Párkeresés futási ideje (ms)"]

    print(tabulate(mydata, headers=headers))

    with open("output/image_matcher_eval.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(mydata, headers=headers))

    cv2.waitKey(0)
