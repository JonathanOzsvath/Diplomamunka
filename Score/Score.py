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

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
minHamming_prefilter = 20
max_correct_radius = 5.0

points = [1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20]


def computeInvHomography(img_ref, img_perspective):
    click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')

    homography_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)
    kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

    matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
    matches = postfilter.ratioFilter(matches, maxRatio=0.8)
    matches = [m for m, n in matches]

    homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches, max_correct_radius=max_correct_radius)

    inv_homography_ransac = np.linalg.inv(homography_ransac)
    inv_homography_ref = np.linalg.inv(homography_ref)

    return inv_homography_ransac, inv_homography_ref


def setMouseCallback(event, x, y, flags, param):
    global click_x, click_y

    if event == cv2.EVENT_LBUTTONUP:
        click_x, click_y = x, y
        cv2.destroyAllWindows()


def calcPointFromPerspectiveToFigure(click_x, click_y, inv_homography_ransac, inv_homography_ref):
    point_in_ref = ime.Project([(click_x, click_y)], inv_homography_ransac)
    point_in_figure = ime.Project(point_in_ref, inv_homography_ref)
    return point_in_figure


def calcScore(point):
    rays = [170, 160, 105, 95, 15.9, 7]
    sector = 0
    multiplier = 1

    d = ime.calcDistance(point, (0, 0))

    point_unit_vector = (point[0] / d, point[1] / d)
    unit_vector = (np.float64(math.cos(math.radians(-81))), np.float64(math.sin(math.radians(-81))))

    dot_product = point_unit_vector[0] * unit_vector[0] + point_unit_vector[1] * unit_vector[1]
    alpha = math.degrees(math.acos(dot_product))

    new_point = (np.float64(point[0] * math.cos(math.radians(-9)) - point[1] * math.sin(math.radians(-9))),
                 np.float64(point[0] * math.sin(math.radians(-9)) - point[1] * math.cos(math.radians(-9))))

    if new_point[0] < 0:
        alpha = 360 - alpha

    sector = points[int(alpha/18)]

    if d <= rays[5]:
        sector = 50
    elif d <= rays[4]:
        sector = 25
    elif d <= rays[2] and d > rays[3]:
        multiplier = 3
    elif d <= rays[0] and d > rays[1]:
        multiplier = 2
    elif d > rays[0]:
        sector = 0
    else:
        multiplier = 1

    score = multiplier * sector
    return score


if __name__ == '__main__':
    name_ref = "video_ref"
    path_ref = '../images/' + name_ref + '.jpg'

    name_perspective = 'video1_0_frame'
    path_perspective = '../images/' + name_perspective + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img_perspective = cv2.imread(path_perspective)
    img_perspective = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

    width = img_ref.shape[1]
    height = img_ref.shape[0]

    cv2.imshow(name_perspective, img_perspective)

    cv2.setMouseCallback(name_perspective, setMouseCallback)
    cv2.waitKey(0)

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    inv_homography_ransac, inv_homography_ref = computeInvHomography(img_ref, img_perspective)

    point_in_figure = calcPointFromPerspectiveToFigure(click_x, click_y, inv_homography_ransac, inv_homography_ref)
    point_in_figure = point_in_figure[0]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = 255
    img = dart_board.drawDartBoard(img, refPoints, circlePoints, numberOfCirclePointPerSector, shift=(width / 2, height / 2), showImage=False)

    score = calcScore(point_in_figure)

    print('Score: {}'.format(score))

    img = cv2.circle(img, (int(point_in_figure[0] + width / 2), int(point_in_figure[1] + height / 2)), 5, (255, 0, 0), thickness=-1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
