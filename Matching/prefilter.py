import cv2
import numpy as np


def prefilter(kp, des, min_hamming):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=2)

    matches = list(filter(lambda m: m[1].distance > min_hamming, matches))

    filtered_indices = [match[0].queryIdx for match in matches]

    kp = [kp[i] for i in filtered_indices]
    des = [des[i] for i in filtered_indices]

    return np.asarray(kp), np.asarray(des)
