import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import ArrowSets


if __name__ == '__main__':
    name = "darts_with_arrow"
    path_arrow = 'output/' + name + '_notArrow.jpg'

    img_arrow = cv2.imread(path_arrow)

    cv2.imshow('arrow', img_arrow)

    L, a, b = cv2.split(img_arrow)

    plt.hist(a.ravel(), 256, [0, 256], label='a')
    # plt.hist(b.ravel(), 256, [0, 256], label='b')

    plt.axis([0, 256, 0, 60000])

    plt.show()

    cv2.waitKey(0)
