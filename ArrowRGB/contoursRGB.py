import os
import cv2
import math
import numpy as np
import RANSAC as ransac
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import dart_board
import cutOutside

numberOfKeypoint = 1000
minHamming_prefilter = 20
numberOfCirclePointPerSector = 10

def calcDistance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


if __name__ == '__main__':
    name = 'darts_with_arrow12_probability_0.3'
    path = '../images/' + name + '.jpg'
    img = cv2.imread(path)

    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspective = 'darts_with_arrow12'
    path_perspective = '../images/' + name_perspective + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    homography_ransac, mask_ransac = ransac.ransac(kp_ref, kp_perspective, matches)

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)
    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')
    homography_matrix_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

    midpoint_ref = ime.Project([(0, 0)], homography_matrix_ref)
    midpoint_perspective = ime.Project(midpoint_ref, homography_ransac)[0]

    circlePoints_ref = ime.Project(circlePoints, homography_matrix_ref)
    refPoints_ref = ime.Project(refPoints, homography_matrix_ref)

    ransac_circlePoints = ime.Project(circlePoints_ref, homography_ransac)
    ransac_refPoints = ime.Project(refPoints_ref, homography_ransac)

    dart_board.drawDartBoard(img, ransac_refPoints, ransac_circlePoints, numberOfCirclePointPerSector, (0, 255, 0),
                             savePath="output/" + name_perspective + "_homography_model_to_image.png")

    # ------------
    # # img1 = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=11)
    # # img1 = cv2.GaussianBlur(img1, (3, 3), 2)
    # # img1 = cv2.Laplacian(img1, cv2.CV_8U)
    # # img1 = cv2.medianBlur(img1, 11)
    # # img1 = cv2.Canny(img1, 30, 200)
    #
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # # dilated = cv2.dilate(img1, kernel)
    #
    # # cv2.imshow('blur', img1)
    #
    # img2 = img1.copy()
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    #
    width = img_ref.shape[1]
    height = img_ref.shape[0]

    inv_homography_ransac = np.linalg.inv(homography_ransac)
    img1 = cv2.warpPerspective(img, inv_homography_ransac, (width, height))
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    cv2.imshow("warp", img1)

    tableBoarderPoint = (225 * math.cos(math.radians(0)), 225 * math.sin(math.radians(0)))
    tableBoarderPoint_ref = ime.Project([tableBoarderPoint], homography_matrix_ref)[0]
    tableOutsideRadious = calcDistance(midpoint_ref[0], tableBoarderPoint_ref)
    img1 = cutOutside.cutOutside(img1, midpoint_ref[0], tableOutsideRadious)
    cv2.imshow("warpCut", img1)

    img1 = cv2.warpPerspective(img1, homography_ransac, (width, height))
    cv2.imshow("warpCutwarp", img1)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     cv2.drawContours(img2, [c], 0, (0, 0, 255), 3)
    #     cv2.imshow("contours", img2)
    #     cv2.waitKey(0)

    for c in contours:
        area = cv2.contourArea(c)

        cv2.circle(img, midpoint_perspective, 3, (255, 0, 0), -1)
        cv2.imshow("contours", img)

        hull = cv2.convexHull(c)
        cv2.drawContours(img, [hull], 0, (0, 0, 255), 1)
        cv2.imwrite("output/" + name_perspective + "_kontur.png", img)

        # if 1000 < area < 1500:
        #     hull = cv2.convexHull(c)
        #     cv2.drawContours(img, [hull], 0, (0, 0, 255), 1)
        #     cv2.imshow("contours", img)
        #     cv2.imwrite("output/" + name_perspective + "_kontur_szurt.png", img)

    cv2.waitKey(0)
