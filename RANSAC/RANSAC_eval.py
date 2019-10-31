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
import pandas as pd
import matplotlib.pyplot as plt

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3

minHamming_prefilter = 20


def gtEvaluate(matches, kp_perspective, truth_points, max_correct_radius=5.0):
    inliers_outlier_mask = []

    for index, match in enumerate(matches):
        d = ime.calcDistance(kp_perspective[match.trainIdx].pt, truth_points[index])
        if d <= max_correct_radius:
            inliers_outlier_mask.append(1)
        else:
            inliers_outlier_mask.append(0)

    return inliers_outlier_mask


def saveEvaluateRansacTable(TP, TN, FP, FN, precision, recall):
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    headers = ["", "GT+", "GT-"]
    data = [["RANSAC+", TP, FP], ["RANSAC-", TN, FN]]

    print(tabulate(data, headers=headers, tablefmt="presto"))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))

    with open("output/RANSAC_TP_TN_FP_FN.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

        f.write('\nprecision: {}\n'.format(precision))
        f.write('recall: {}\n'.format(recall))


def saveRansacTable(data, name):
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    headers = ["Image", "MaxRadius", "#Matches", "#Inlier", "TP", "TN", "FP", "FN", "%Precision", "%Recall", "Runtime(ms)"]

    print(tabulate(data, headers=headers, tablefmt="presto"))
    with open("output/" + name + ".txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

    return headers


def addTableRaw(data, name_perspective, max_correct_radius, matches, inlier, TP, TN, FP, FN, precision, recall, runtime):
    data.append([name_perspective, max_correct_radius, matches, inlier, TP, TN, FP, FN, precision, recall, runtime])
    return data


def evaluateRansac(mask_inliers_outlier, mask_ransac):
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
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    return TP, TN, FP, FN, precision * 100, recall * 100


def runEval(img_ref, name_perspective, path_perspective, data, max_correct_radius=5.0, drawGrid=False):
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
    mask_inliers_outlier = gtEvaluate(matches, kp_perspective, truth_points, max_correct_radius)

    start = time.time()
    homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches, max_correct_radius=max_correct_radius)
    mask_ransac = [m[0] for m in mask_ransac]
    end = time.time()
    ransacRunTime = ime.getRunTime(start, end)

    TP, TN, FP, FN, precision, recall = evaluateRansac(mask_inliers_outlier, mask_ransac)
    data = addTableRaw(data, name_perspective, max_correct_radius, len(matches), len([i for i in mask_ransac if i == 1]), TP, TN, FP, FN, precision, recall, ransacRunTime)

    if drawGrid:
        circlePoints_ref = ime.Project(circlePoints, homography_matrix_ref)
        ransac_ciclePoints = ime.Project(circlePoints_ref, homography_ransac)
        ime.drawPoints(img_perspective, ransac_ciclePoints, name_perspective)

    return data


def addAverageRow(average, average_tmp):
    average.append(["Average", average_tmp[0][1],
                        round(statistics.mean([i[2] for i in average_tmp]), 2),
                        round(statistics.mean([i[3] for i in average_tmp]), 2),
                        round(statistics.mean([i[4] for i in average_tmp]), 2),
                        round(statistics.mean([i[5] for i in average_tmp]), 2),
                        round(statistics.mean([i[6] for i in average_tmp]), 2),
                        round(statistics.mean([i[7] for i in average_tmp]), 2),
                        round(statistics.mean([i[8] for i in average_tmp]), 2),
                        round(statistics.mean([i[9] for i in average_tmp]), 2),
                        round(statistics.mean([float(i[10]) for i in average_tmp]), 2)])

    return average


def saveAveragePlot(average, headers):
    data2 = pd.DataFrame(average, columns=headers)

    x = max_correct_radius
    y_precision = [data2.groupby("MaxRadius").get_group(i).values[0][8] for i in max_correct_radius]
    y_recall = [data2.groupby("MaxRadius").get_group(i).values[0][9] for i in max_correct_radius]

    line_precision = plt.plot(x, y_precision, 'g-', x, y_precision, 'g^')
    line_recall = plt.plot(x, y_recall, 'b-', x, y_recall, 'b+')

    plt.xlabel('Max Correct Radious')
    plt.ylabel('%')

    legends = ['Precisoin', 'Recall']
    plt.legend([line_precision[0], line_recall[0]], legends)
    plt.savefig('output/Precision_Recall.png', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    data = []
    average = []
    average_tmp = []
    max_correct_radius = range(10)

    name_ref = "darts1_1"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.resize(img_ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    for r in max_correct_radius:
        for name_perspective in name_perspectives:
            path_perspective = '../images/' + name_perspective + '.jpg'
            data = runEval(img_ref, name_perspective, path_perspective, data, r, drawGrid=False)
            average_tmp.append(data[-1])
        average = addAverageRow(average, average_tmp)
        average_tmp.clear()


    saveRansacTable(data=data, name="RANSAC_eval")
    headers = saveRansacTable(data=average, name="RANSAC_average_eval")
    saveAveragePlot(average, headers)

    cv2.waitKey(0)
