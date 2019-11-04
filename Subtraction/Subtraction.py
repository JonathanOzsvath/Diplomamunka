import os
import cv2
import math
import numpy as np
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import dart_board
import RANSAC
import math

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
minHamming_prefilter = 20
max_correct_radius = 5.0

if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    # name_perspective = 'darts_bal'
    name_perspective = 'darts_with_arrow'
    path_perspective = '../images/' + name_perspective + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    width = img_ref.shape[1]
    height = img_ref.shape[0]

    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')
    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)
    homography_matrix_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches, max_correct_radius=max_correct_radius)
    inv_homography_ransac = np.linalg.inv(homography_ransac)

    middlePoint_ref = ime.Project([(0, 0)], homography_matrix_ref)

    img = cv2.warpPerspective(img_perspective, inv_homography_ransac, (width, height))
    # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Ref', img_ref)
    cv2.imwrite("output/" + name_ref + "_ref.jpg", img_ref)

    cv2.imshow('Perspective', img)
    cv2.imwrite("output/" + name_perspective + "_perspective.jpg", img)

    img_ref = img_ref[:] / 255
    img = img[:] / 255

    img_subtracted = abs(img_ref - img)

    img_subtracted = img[:] * 255

    cv2.imshow('subtracted', img_subtracted)
    cv2.imwrite("output/" + name_perspective + "_subtracted.jpg", img_subtracted)

    cv2.waitKey(0)
