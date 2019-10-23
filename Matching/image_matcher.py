import cv2
import ownBF


def BF(kp_ref, des_ref, kp_perspective, des_perspective):
    firstTwoArray = ownBF.bruteForce(kp_ref, des_ref, kp_perspective, des_perspective)
    return firstTwoArray


def openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False):
    if crossCheck:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des_ref, des_perspective)
        matches = [[match] for match in matches]

    else:
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Match descriptors.
        matches = bf.knnMatch(des_ref, des_perspective, k=2)

    return matches


def flann(des_ref, des_perspective):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,  # 12
                        key_size=20,  # 20
                        multi_probe_level=2)  # 2

    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_ref, des_perspective, k=2)

    matches = list(filter(lambda m: len(m) == 2, matches))

    return matches
