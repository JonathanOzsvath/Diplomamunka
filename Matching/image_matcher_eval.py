import os
import cv2
import time
import numpy as np
import image_matcher as im
from tabulate import tabulate
import dart_board

numberOfKeypoint = 500
numberOfCirclePointPerSector = 3


def getRunTime(start, end):
    return str(end - start)


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


def evaluate(matches, kp_ref, kp_perspective):
    # TODO
    pass


if __name__ == '__main__':
    name_ref = '../images/darts1_1.jpg'
    name_perspective = '../images/darts1_2.jpg'
    crossCheck = False

    img_ref = cv2.imread(name_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(name_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    click_point_ref = LoadPoints(os.path.splitext(name_ref)[0])
    click_point_perspective = LoadPoints(os.path.splitext(name_perspective)[0])

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)
    # TODO homográfiák számolása

    # Initiate ORB
    start = time.time()
    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)
    end = time.time()

    detectRunTime = getRunTime(start, end)

    # index_ref, index_perspective = \
    # im.BF(kp_ref, des_ref, kp_perspective, des_perspective)

    start = time.time()
    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=crossCheck)
    end = time.time()
    matchingRunTime = getRunTime(start, end)

    mydata = [(
        name_ref, "-", str(numberOfKeypoint), "-", name_perspective, str(numberOfKeypoint), detectRunTime,
        "cvBF_noprefilter_nocheck_nohamming_noratio",
        crossCheck, "-",
        "-", "-", "-",
        "-", "-", matchingRunTime)]

    headers = ["Referenciakép neve", "Előszűrés küszöbértéke", "Detektált pontok száma", "Szűrt pontok száma",
               "Perspektív kép neve", "Detektált pontok száma perspektív képen", "Detektálás futási ideje (ms)",
               "Párkeresési módszer", "Cross check", "Távolság utószűrés paramétere", "Arány utószűrés paramétere",
               "Megtalált párok száma", "Párosított pontok %-a", "Helyes párok száma", "Helyes párok %-a",
               "Párkeresés futási ideje (ms)"]

    print(tabulate(mydata, headers=headers))
