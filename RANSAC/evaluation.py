import os
import cv2
import time
import math
import statistics
import numpy as np
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import dart_board

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3

minHamming_prefilter = 20
max_correct_radius = 5.0


# def calcDistance(point1, point2):
#     return math.sqrt(
#         (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))
#
#
# def evaluate(matches, kp_perspective, truth_points):
#     inliers_match_index = []
#
#     for index, match in enumerate(matches):
#         d = calcDistance(kp_perspective[match[0].trainIdx].pt, truth_points[index])
#         if d <= max_correct_radius:
#             inliers_match_index.append(index)
#         else:
#             outliers_match_index.append(index)
#
#     return inliers_outlier_mask


def gtFindHomography():
    # TODO hasonlóan mint a ransac vissza adja a homography, mask, => átgondolni, hogy a match hova lerüljön
    pass


if __name__ == '__main__':
    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspective = 'darts2_1'
    path_perspective = '../images/' + name_perspective + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.resize(img_perspective, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')
    click_point_perspective = ime.LoadPoints(os.path.splitext(path_perspective)[0] + '.click')

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    homography_matrix_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))
    homography_matrix_perspective, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_perspective))
    homography_matrix_ground_truth = np.dot(homography_matrix_perspective, np.linalg.inv(homography_matrix_ref))

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)

    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)

    # truth_points = ime.getMatchesPointWithHomography(kp_ref, matches, homography_matrix_ground_truth)
