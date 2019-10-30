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

numberOfKeypoint = 1000

minHamming_prefilter = 20


def ransac(kp_ref, kp_perspective, matches, max_correct_radius=5.0, min_match_count=10):

    if len(matches) > min_match_count:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_perspective[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # mask-ban az 1-es jelöli az inliereket
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, max_correct_radius)
        a = 0
    else:
        homography = False
        mask = False
        print("Not enough matches are found - %d/%d".format(len(matches), min_match_count))

    return homography, mask


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

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    homography, mask = ransac(kp_ref, kp_perspective, matches)

    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')

    ime.drawPoints(img_perspective, ime.Project(click_point_ref, homography), 'click')

    cv2.waitKey(0)
