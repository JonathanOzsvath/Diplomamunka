import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

numberOfKeypoint = 100


def functionSwitcher(argument):
    # Get the function from switcher dictionary
    func = switcher.get(argument, "nothing")
    # Execute the function
    return func()


def drawMatches(perspectiveImage, matches, kpReference, kpPerspective, fileName, isGray=True):
    if isGray:
        perspectiveImage = cv2.cvtColor(perspectiveImage, cv2.COLOR_GRAY2BGR)

    for i in matches:
        image1 = cv2.circle(perspectiveImage, (int(kpPerspective[i.trainIdx].pt[0]), int(kpPerspective[i.trainIdx].pt[1])),
                            int(kpPerspective[i.trainIdx].size / 2), (0, 255, 0), thickness=1)
        image1 = cv2.line(image1, (int(kpReference[i.queryIdx].pt[0]), int(kpReference[i.queryIdx].pt[1])),
                          (int(kpPerspective[i.trainIdx].pt[0]), int(kpPerspective[i.trainIdx].pt[1])), color=(0, 255, 255), thickness=2)

    cv2.imshow(fileName, image1)
    # cv2.imwrite("output/" + fileName + ".jpg", image1)


def draw2NNMatches(perspectiveImage, matches, kpReference, kpPerspective, fileName, isGray=True):
    if isGray:
        perspectiveImage = cv2.cvtColor(perspectiveImage, cv2.COLOR_GRAY2BGR)

    for m, n in matches:
        image1 = cv2.circle(perspectiveImage, (int(kpPerspective[m.trainIdx].pt[0]), int(kpPerspective[m.trainIdx].pt[1])),
                            int(kpPerspective[m.trainIdx].size / 2), (0, 255, 0), thickness=1)
        image1 = cv2.line(image1, (int(kpReference[m.queryIdx].pt[0]), int(kpReference[m.queryIdx].pt[1])),
                          (int(kpPerspective[m.trainIdx].pt[0]), int(kpPerspective[m.trainIdx].pt[1])), color=(0, 255, 255), thickness=2)

    # cv2.imshow(fileName, image1)
    cv2.imwrite("output/" + fileName + ".jpg", image1)


def bfWithOutCrossCheck():
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    drawMatches(img2, matches[:], kpReference=kp1, kpPerspective=kp2, fileName="bfWithOutCrossCheck", isGray=True)


def bfWithCrossCheck():
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    drawMatches(img2, matches[:], kpReference=kp1, kpPerspective=kp2, fileName="bfWithCrossCheck", isGray=True)


def bfKnnMatcher():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good.append([m])

    draw2NNMatches(img2, matches[:], kpReference=kp1, kpPerspective=kp2, fileName="bfKnnMatcher_0_5", isGray=True)


switcher = {
    0: bfWithOutCrossCheck,
    1: bfWithCrossCheck,
    2: bfKnnMatcher
}

if __name__ == '__main__':
    # referencia kép
    img = cv2.imread('../images/darts1_2.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perspektív kép
    img = cv2.imread('../images/darts2_2.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate ORB
    orb = cv2.ORB_create(numberOfKeypoint)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    img_keyPoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_keyPoints2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("img_keyPoints", img_keyPoints)
    # cv2.imwrite("output/img_keyPoints.jpg", img_keyPoints)
    # cv2.imshow("img_keyPoints2", img_keyPoints2)
    # cv2.imwrite("output/img_keyPoints2.jpg", img_keyPoints2)

    # functionSwitcher(0)
    # functionSwitcher(1)
    functionSwitcher(2)

    cv2.waitKey(0)
