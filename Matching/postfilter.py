import cv2
import numpy as np


def distanceFilter(matches, maxHamming):
    matches = list(filter(lambda m: m[0].distance < maxHamming, matches))

    return matches


def ratioFilter(matches, maxRatio):
    matches = list(filter(lambda m: m[0].distance / m[1].distance <= maxRatio, matches))

    return matches
