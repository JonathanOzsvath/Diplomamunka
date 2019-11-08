import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import ArrowSets


def segment(a_good, b_good, img_yuv):

    height, width = img_yuv.shape[:2]
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            a = img_yuv[y, x][1]
            b = img_yuv[y, x][2]
            if b in b_good and a in a_good:
                img[y, x] = 255

    cv2.imshow("segment", img)
    # cv2.imwrite("output/" + name + "_segment.jpg", img)


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = "darts_with_arrow3"
    path = '../images/' + name + '.jpg'

    img = cv2.imread(path)
    img_resize = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

    arrows, notArrows = ArrowSets.loadArrowSets()

    arrows_a = [i[0] for i in arrows]
    arrows_b = [i[1] for i in arrows]

    notArrows_a = [i[0] for i in notArrows]
    notArrows_b = [i[1] for i in notArrows]
    # cv2.imshow('YUV', img_yuv)

    training_arrow = [[arrows_a[i], arrows_b[i]] for i in range(len(arrows_a))]
    training_notArrow = [[notArrows_a[i], notArrows_b[i]] for i in range(len(notArrows_a))]

    labels = np.array([1] * len(arrows_a) + [-1] * len(notArrows_a))
    trainingData = np.array(training_arrow + training_notArrow, dtype=np.float32)

    # Train the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)
    svm.save("output/svm")

    # svm = cv2.ml.SVM_load("output/svm")

    a_good = []
    b_good = []

    a_bad = []
    b_bad = []

    green = (0, 255, 0)
    blue = (255, 0, 0)
    for i in range(256):
        for j in range(256):
            sampleMat = np.array([[i, j]], dtype=np.float32)
            response = svm.predict(sampleMat)[1]
            if response == 1:
                a_good.append(i)
                b_good.append(j)
            elif response == -1:
                a_bad.append(i)
                b_bad.append(j)

    # plt.plot(arrows_b, arrows_a, 'gx')
    # plt.plot(notArrows_b, notArrows_a, 'rx')
    plt.plot(a_bad, b_bad, 'ro')
    plt.plot(a_good, b_good, 'go')

    plt.xlabel('V')
    plt.ylabel('U')
    plt.axis([0, 256, 0, 256])

    plt.show()

    # img_resize = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    # img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

    segment(a_good, b_good, img_yuv)

    cv2.waitKey(0)