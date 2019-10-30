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
import RANSAC
from tabulate import tabulate

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3

minHamming_prefilter = 20
max_correct_radius = 5.0


def gtEvaluate(matches, kp_perspective, truth_points):
    inliers_outlier_mask = []

    for index, match in enumerate(matches):
        d = ime.calcDistance(kp_perspective[match.trainIdx].pt, truth_points[index])
        if d <= max_correct_radius:
            inliers_outlier_mask.append(1)
        else:
            inliers_outlier_mask.append(0)

    return inliers_outlier_mask


def saveEvaluateRansacTable(TP, TN, FP, FN, precision, recall):
    headers = ["", "GT+", "GT-"]
    data = [["RANSAC+", TP, FP], ["RANSAC-", TN, FN]]

    print(tabulate(data, headers=headers, tablefmt="presto"))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))

    with open("output/RANSAC_TP_TN_FP_FN.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

        f.write('\nprecision: {}\n'.format(precision))
        f.write('recall: {}\n'.format(recall))


def saveRansacTable(data):
    headers = ["Image", "#Matches", "#Inlier", "TP", "TN", "FP", "FN", "%Precision", "%Recall", "Runtime(ms)"]

    print(tabulate(data, headers=headers, tablefmt="presto"))
    with open("output/RANSAC_eval.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))


def addTableRaw(data, name_perspective, matches, inlier, TP, TN, FP, FN, precision, recall, runtime):
    data.append([name_perspective, matches, inlier, TP, TN, FP, FN, precision, recall, runtime])
    return data


def evaluateRansac(mask_inliers_outlier, mask_ransac):
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(mask_inliers_outlier)):
        if mask_inliers_outlier[i] == 1 and mask_ransac[i] == 1:
            TP += 1
        elif mask_inliers_outlier[i] == 0 and mask_ransac[i] == 1:
            FP += 1
        elif mask_inliers_outlier[i] == 0 and mask_ransac[i] == 0:
            TN += 1
        elif mask_inliers_outlier[i] == 1 and mask_ransac[i] == 0:
            FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return TP, TN, FP, FN, precision, recall


def runEval(img_ref, path_perspective, data):
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
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    truth_points = ime.getMatchesPointWithHomography(kp_ref, matches, homography_matrix_ground_truth)
    mask_inliers_outlier = gtEvaluate(matches, kp_perspective, truth_points)

    start = time.time()
    homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches)
    mask_ransac = [m[0] for m in mask_ransac]
    end = time.time()
    ransacRunTime = ime.getRunTime(start, end)

    TP, TN, FP, FN, precision, recall = evaluateRansac(mask_inliers_outlier, mask_ransac)
    data = addTableRaw(data, name_perspective, len(matches), len([i for i in mask_ransac if i == 1]), TP, TN, FP, FN, precision, recall, ransacRunTime)

    return data


if __name__ == '__main__':
    data = []

    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    for name_perspective in name_perspectives:
        path_perspective = '../images/' + name_perspective + '.jpg'
        data = runEval(img_ref, path_perspective, data)

    saveRansacTable(data=data)
