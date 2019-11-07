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
import cutOutside as co

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
minHamming_prefilter = 20
max_correct_radius = 5.0


def saveArrowSets(arrow, notArrow):
    with open('output/arrowSet.txt', 'a') as f:
        for x, y in arrow:
            f.write('{}, {}\n'.format(x, y))

    with open('output/notArrowSet.txt', 'a') as f:
        for x, y in notArrow:
            f.write('{}, {}\n'.format(x, y))


def loadArrowSets():
    arrow = []
    notArrow = []

    with open('output/arrowSet.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            arrow.append((int(x), int(y)))

    with open('output/notArrowSet.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            notArrow.append((int(x), int(y)))

    return arrow, notArrow


def calcDistance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


def computeHomography(img_ref, img_perspective):
    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)
    homography_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches, max_correct_radius=max_correct_radius)

    return homography_ransac, homography_ref


def getArrow_notArrow(name, cutImg, cutImg_Lab, maskImg):
    height, width = cutImg.shape[:2]
    img_arrow = np.zeros(cutImg.shape, dtype=np.uint8)
    img_notArrow = np.zeros(cutImg.shape, dtype=np.uint8)
    arrow = []
    notArrow = []

    for y in range(0, height):
        for x in range(0, width):
            if maskImg[y, x] == 255 and set(cutImg[y, x]) != {255, 255, 255}:
                arrow.append(cutImg_Lab[y, x][1:])
                img_arrow[y, x] = cutImg_Lab[y, x]
            elif maskImg[y, x] == 0 and set(cutImg[y, x]) != {255, 255, 255}:
                notArrow.append(cutImg_Lab[y, x][1:])
                img_notArrow[y, x] = cutImg_Lab[y, x]

    # cv2.imwrite("output/" + name + "_arrow.jpg", img_arrow)
    # cv2.imwrite("output/" + name + "_notArrow.jpg", img_notArrow)

    return arrow, notArrow


def getImages(img_ref_gray, path_perspective, path_perspective_mask):
    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective_gray = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    img_perspective_mask = cv2.imread(path_perspective_mask, 0)
    img_perspective_mask = cv2.resize(img_perspective_mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    homography_ransac, homography_ref = computeHomography(img_ref_gray, img_perspective_gray)

    midPoint_ref = ime.Project([(0, 0)], homography_ref)[0]

    tableBoarderPoint = (225 * math.cos(math.radians(0)), 225 * math.sin(math.radians(0)))
    tableBoarderPoint_ref = ime.Project([tableBoarderPoint], homography_ref)[0]

    tableOutsideRadious = calcDistance(midPoint_ref, tableBoarderPoint_ref)

    inv_homography_ransac = np.linalg.inv(homography_ransac)

    height, width = img_perspective.shape[:2]
    img = cv2.warpPerspective(img_perspective, inv_homography_ransac, (width, height))

    cutImg = co.cutOutside(img, midPoint_ref, tableOutsideRadious)
    # cv2.imshow("cut", cutImg)

    # cutImg_Lab = cv2.cvtColor(cutImg, cv2.COLOR_BGR2Lab)
    cutImg_Lab = cv2.cvtColor(cutImg, cv2.COLOR_BGR2YUV)
    # cv2.imshow("cutLab", cutImg_Lab)

    img_perspective_mask = cv2.warpPerspective(img_perspective_mask, inv_homography_ransac, (width, height))
    # cv2.imshow("mask", img_perspective_mask)

    return cutImg, cutImg_Lab, img_perspective_mask


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filePath = "output/arrowSet.txt"
    if os.path.exists(filePath):
        os.remove(filePath)

    filePath = "output/notArrowSet.txt"
    if os.path.exists(filePath):
        os.remove(filePath)

    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspectives = ['darts_with_arrow', 'darts_with_arrow2', 'darts_with_arrow3', 'darts_with_arrow4', 'darts_with_arrow5', 'darts_with_arrow6',
                         'darts_with_arrow7', 'darts_with_arrow8', 'darts_with_arrow9', 'darts_with_arrow10']
    # name_perspectives = ['darts_with_arrow', 'darts_with_arrow2']

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    for name_perspective in name_perspectives:
        path_perspective = '../images/' + name_perspective + '.jpg'

        name_perspective_mask = name_perspective + '_mask'
        path_perspective_mask = '../images/' + name_perspective_mask + '.jpg'

        cutImg, cutImg_Lab, img_perspective_mask = getImages(img_ref_gray, path_perspective, path_perspective_mask)

        arrow, notArrow = getArrow_notArrow(name_perspective, cutImg, cutImg_Lab, img_perspective_mask)

        saveArrowSets(arrow, notArrow)

    cv2.waitKey(0)
