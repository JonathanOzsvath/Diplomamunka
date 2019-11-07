import os
import cv2
import math
import numpy as np
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import RANSAC
import dart_board

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
minHamming_prefilter = 20
max_correct_radius = 5.0


def calcDistance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def cutOutside(img, midPoint, tableOutsideRadious):
    height, width = img.shape[:2]
    for y in range(0, height):
        for x in range(0, width):
            if calcDistance((x, y), midPoint) > tableOutsideRadious:
                img[y, x] = (255, 255, 255)

    return img


if __name__ == '__main__':
    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspective = 'darts_with_arrow'
    path_perspective = '../images/' + name_perspective + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)
    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')

    homography_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

    midPoint_ref = ime.Project([(0, 0)], homography_ref)[0]

    tableBoarderPoint = (225 * math.cos(math.radians(0)), 225 * math.sin(math.radians(0)))
    tableBoarderPoint_ref = ime.Project([tableBoarderPoint], homography_ref)[0]

    tableOutsideRadious = calcDistance(midPoint_ref, tableBoarderPoint_ref)

    inv_homography_ref = np.linalg.inv(homography_ref)

    cutImg = cutOutside(img_ref, midPoint_ref, tableOutsideRadious)
    cv2.imshow("cut", cutImg)

    cv2.waitKey(0)
