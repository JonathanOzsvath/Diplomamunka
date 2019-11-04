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


def save(directoryName, img1, img1_name, img2, img2_name, img_subtracted, ifShow=False):
    directory = "output/" + directoryName + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(directory + img1_name + '.jpg', img1)
    cv2.imwrite(directory + img2_name + "subtracted.jpg", img_subtracted)
    cv2.imwrite(directory + img2_name + '_warpPerspective.jpg', img2)

    if ifShow:
        cv2.imshow(img1_name, img1)
        cv2.imshow(img2_name, img2)
        cv2.imshow('subtracted', img_subtracted)


def withoutFilter(img1, img1_name, img2, img2_name):
    img_subtracted = subtraction(img1, img2)

    save('', img1, img1_name, img2, img2_name, img_subtracted)


def averaging(img1, img1_name, img2, img2_name):
    img1_blur = cv2.blur(img1, (5, 5))
    img2_blur = cv2.blur(img2, (5, 5))
    img_subtracted = subtraction(img1_blur, img2_blur)

    save('Averaging', img1_blur, img1_name, img2_blur, img2_name, img_subtracted)


def gaussian(img1, img1_name, img2, img2_name):
    img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
    img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)
    img_subtracted = subtraction(img1_blur, img2_blur)

    save('Gaussian', img1_blur, img1_name, img2_blur, img2_name, img_subtracted)


def median(img1, img1_name, img2, img2_name):
    img1_blur = cv2.medianBlur(img1, 5)
    img2_blur = cv2.medianBlur(img2, 5)
    img_subtracted = subtraction(img1_blur, img2_blur)

    save('Median', img1_blur, img1_name, img2_blur, img2_name, img_subtracted)


def subtraction(img1, img2):
    img1 = img1[:] / 255
    img2 = img2[:] / 255

    img_subtracted = abs(img1 - img2)

    img_subtracted = np.array([[np.uint8(j * 255) for j in i] for i in img_subtracted])

    return img_subtracted


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

    img = cv2.warpPerspective(img_perspective, inv_homography_ransac, (width, height))

    withoutFilter(img_ref, name_ref, img, name_perspective)
    averaging(img_ref, name_ref, img, name_perspective)
    gaussian(img_ref, name_ref, img, name_perspective)
    median(img_ref, name_ref, img, name_perspective)

    cv2.waitKey(0)
