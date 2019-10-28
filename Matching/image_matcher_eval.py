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
    return str(round((end - start) * 1000, 2))


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
    # points = [(index, kp_ref[match[0].queryIdx].pt) for index, match in enumerate(matches)]
    points = [kp_ref[match[0].queryIdx].pt for match in matches]
    return Project(points, homography_matrix)


def evaluate(matches, kp_perspective, truth_points):
    inliers_match_index = []
    outliers_match_index = []

    for index, match in enumerate(matches):
        d = calcDistance(kp_perspective[match[0].trainIdx].pt, truth_points[index])
        if d <= max_correct_radius:
            inliers_match_index.append(index)
        else:
            outliers_match_index.append(index)

    return inliers_match_index, outliers_match_index


def drawGT(img, click_point_ref, homography_matrix_ground_truth, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    projected_click_point_ref = Project(click_point_ref, homography_matrix_ground_truth)

    for point in projected_click_point_ref:
        img = cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), thickness=-1)

    cv2.imwrite(directory + '/gt_' + name_perspective + '.jpg', img)


def drawOrb(img, kp_perspective, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for kp in kp_perspective:
        img = cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), thickness=-1)

    cv2.imwrite(directory + '/orb_' + name_perspective + '.jpg', img)


def drawMatched(img, truth_points, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for point in truth_points:
        img = cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), thickness=-1)

    cv2.imwrite(directory + '/matched_' + name_perspective + '.jpg', img)


def drawMatching(img, kp_ref, kp_perspective, matches, inliers_match_index, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in inliers_match_index:
        point1 = kp_ref[matches[i][0].queryIdx].pt
        point2 = kp_perspective[matches[i][0].trainIdx].pt
        img = cv2.circle(img, (int(point1[0]), int(point1[1])), 2, (0, 255, 0), thickness=-1)
        img = cv2.circle(img, (int(point2[0]), int(point2[1])), 2, (0, 255, 0), thickness=-1)

        img = cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), thickness=2)

    cv2.imwrite(directory + '/matching_' + name_perspective + '.jpg', img)


def drawEval(img, kp_perspective, matches, inliers_match_index, outliers_match_index, truth_points, outputFolderName, isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    directory = "output/" + outputFolderName
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in outliers_match_index:
        point1 = kp_perspective[matches[i][0].trainIdx].pt
        point2 = truth_points[i]
        img = cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), thickness=2)
        img = cv2.circle(img, (int(point1[0]), int(point1[1])), 2, (255, 0, 0), thickness=-1)

    for i in inliers_match_index:
        point1 = kp_perspective[matches[i][0].trainIdx].pt
        point2 = truth_points[i]
        img = cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 255), thickness=2)
        img = cv2.circle(img, (int(point1[0]), int(point1[1])), 2, (0, 255, 0), thickness=-1)

    cv2.imwrite(directory + '/eval_' + name_perspective + '.jpg', img)


def drawPoints(img, points, name, color=(0, 255, 0), isGray=True):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, (int(point[0]), int(point[1])), 3, color, thickness=-1)

    cv2.imshow(name, img)


def addTableRow(methodId, name_ref, minHammming, numberOfRefKeyPoint, numberOfFilteredKeyPoints, name_perspective, numberOfPerspectiveKeyPoint,
                detectRunTime, methodName, crossCheck, maxHamming_postfilter, maxRatio_postfilter, foundMatches,
                foundMatchesPercent, numberOfInliers, correctMatchPercent, matchingRunTime):
    mydata.append([methodId, name_ref, minHammming, numberOfRefKeyPoint, numberOfFilteredKeyPoints, name_perspective, numberOfPerspectiveKeyPoint,
                   detectRunTime, methodName, crossCheck, maxHamming_postfilter, maxRatio_postfilter, foundMatches,
                   round(foundMatchesPercent, 1), numberOfInliers, round(correctMatchPercent, 1), matchingRunTime])


def runMethod(methodId, method_name, name_ref, name_perspective, prefilterValue, crossCheck, postFilterHamming, postFilterRatio, outputName):
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
    else:
        minHamming_prefilter = '-'

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
    inliers_match_index, outliers_match_index = evaluate(matches, kp_perspective, truth_points)

    # drawOrb(img_perspective, kp_perspective, outputName)
    # drawGT(img_perspective, click_point_ref, homography_matrix_ground_truth, outputName)
    # drawMatched(img_perspective, truth_points, outputName)
    # drawMatching(img_perspective, kp_ref, kp_perspective, matches, inliers_match_index, outputName)
    # drawEval(img_perspective, kp_perspective, matches, inliers_match_index, outliers_match_index,truth_points, outputName)

    numberOfInliers = len(inliers_match_index)
    foundMatches = len(matches)
    numberOfFilteredKeyPoints = len(kp_ref)

    if numberOfFilteredKeyPoints != 0:
        foundMatchesPercent = (foundMatches / numberOfFilteredKeyPoints * 100)
    else:
        foundMatchesPercent = 0

    if foundMatches != 0:
        numberOfInliersPercent = (numberOfInliers / foundMatches * 100)
    else:
        numberOfInliersPercent = 0

    # addTableRow(methodId, name_ref, minHamming_prefilter, numberOfKeypoint, numberOfFilteredKeyPoints, name_perspective,
    #             numberOfKeypoint, detectRunTime, outputName, crossCheck, maxHamming_postfilter, maxRatio_postfilter,
    #             foundMatches, (foundMatches / numberOfFilteredKeyPoints * 100), numberOfInliers,
    #             (numberOfInliers / foundMatches * 100), matchingRunTime)


    summary.append([methodId, name_ref, minHamming_prefilter, numberOfKeypoint, numberOfFilteredKeyPoints, name_perspective,
                    numberOfKeypoint, detectRunTime, outputName, crossCheck, maxHamming_postfilter, maxRatio_postfilter,
                    foundMatches, foundMatchesPercent, numberOfInliers,
                    numberOfInliersPercent, matchingRunTime])


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


def addSummaryRow():
    addTableRow(summary[0][0], summary[0][1], summary[0][2], summary[0][3], round(statistics.mean([float(i[4]) for i in summary]), 2), "Average",
                summary[0][6], round(statistics.mean([float(i[7]) for i in summary]), 2), summary[0][8], summary[0][9], summary[0][10], summary[0][11],
                round(statistics.mean([int(i[12]) for i in summary]), 2), round(statistics.mean([float(i[13]) for i in summary]), 1), round(statistics.mean([float(i[14]) for i in summary]), 2),
                round(statistics.mean([float(i[15]) for i in summary]), 1), round(statistics.mean([float(i[16]) for i in summary]), 2))


if __name__ == '__main__':
    methodId = 1
    name_ref = "darts1_1"

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']
    # name_perspectives = ['darts2_1']
    # minHamming_prefilters = [False, 45, 50, 55]
    minHamming_prefilters = [False]
    maxHamming_postfilters = [False]
    # maxHamming_postfilters = range(0, 255, 5)
    cross_Checks = [False]
    # maxRatio_postfilters = [False, 0.6, 0.7, 0.8]
    maxRatio_postfilters = [1.0]
    methodNames = ['cvBF']

    # name_perspective = "darts2_1"
    # minHamming_prefilter = 60
    # maxHamming_postfilter = 48
    # cross_Check = False
    # maxRatio_postfilter = 0.7
    # methodName = "cvBF"

    # for name_perspective in name_perspectives:
    #     runMethod(methodId, 'BF', name_ref, name_perspective, prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False,
    #               outputName=makeMethodName('BF', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False))
    #     print(makeMethodName('BF', prefilterValue=False, crossCheck=False, postFilterHamming=False, postFilterRatio=False) + '_' + name_perspective)
    #     if len(summary) == len(name_perspectives):
    #         addSummaryRow()
    #         summary.clear()
    #         methodId += 1

    for methodName in methodNames:
        for maxRatio_postfilter in maxRatio_postfilters:
            for cross_Check in cross_Checks:
                if (cross_Check and maxRatio_postfilter) or (methodName == 'FLANN' and cross_Check):
                    continue
                for maxHamming_postfilter in maxHamming_postfilters:
                    for minHamming_prefilter in minHamming_prefilters:
                        for name_perspective in name_perspectives:
                            runMethod(methodId, methodName, name_ref, name_perspective, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter,
                                      outputName=makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter))
                            print(makeMethodName(methodName, prefilterValue=minHamming_prefilter, crossCheck=cross_Check, postFilterHamming=maxHamming_postfilter, postFilterRatio=maxRatio_postfilter) + '_' + name_perspective)

                            if len(summary) == len(name_perspectives):
                                addSummaryRow()
                                summary.clear()
                                methodId += 1

    # headers = ["Referenciakép neve", "Előszűrés küszöbértéke", "Detektált pontok száma", "Szűrt pontok száma",
    #            "Perspektív kép neve", "Detektált pontok száma perspektív képen", "Detektálás futási ideje (ms)",
    #            "Párkeresési módszer", "Cross check", "Távolság utószűrés paramétere", "Arány utószűrés paramétere",
    #            "Megtalált párok száma", "Párosított pontok %-a", "Helyes párok száma", "Helyes párok %-a",
    #            "Párkeresés futási ideje (ms)"]

    headers = ["Id", "Image", "Prefilt", "#Detected", "#Filt",
               "Image", "#Detected", "Det(ms)",
               "Method", "Cross", "Max.Dist.", "Max.Ratio",
               "#Matches", "%Matches", "#Correct", "%Correct",
               "Match(ms)"]

    print(tabulate(mydata, headers=headers))

    with open("output/image_matcher_eval.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(mydata, headers=headers))

    cv2.waitKey(0)
