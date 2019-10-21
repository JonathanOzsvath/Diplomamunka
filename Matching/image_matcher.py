import cv2
import ownBF


def BF(kp_ref, des_ref, kp_perspective, des_perspective):
    firstTwoArray = ownBF.bruteForce(kp_ref, des_ref, kp_perspective, des_perspective)
    return firstTwoArray


def openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    # Match descriptors.
    matches = bf.knnMatch(des_ref, des_perspective, k=2)

    return matches
