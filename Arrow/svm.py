import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import drawArrow


def segment(v_good,u_good, img_yuv):

    height, width = img_yuv.shape[:2]
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            u = img_yuv[y, x][1]
            v = img_yuv[y, x][2]
            if u in u_good and v in v_good:
                img[y, x] = 255

    cv2.imshow("segment", img)
    cv2.imwrite("output/" + name + "_segment.jpg", img)


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = "darts_with_arrow3"
    path = '../images/' + name + '.jpg'

    img = cv2.imread(path)
    img_resize = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

    arrow, notArrow = drawArrow.loadArrowSets()

    u_arrow = [img_yuv[y, x][1] for x, y in arrow]
    v_arrow = [img_yuv[y, x][2] for x, y in arrow]

    u_notArrow = [img_yuv[y, x][1] for x, y in notArrow]
    v_notArrow = [img_yuv[y, x][2] for x, y in notArrow]

    # cv2.imshow('YUV', img_yuv)

    training_arrow = [[v_arrow[i], u_arrow[i]] for i in range(len(u_arrow))]
    training_notArrow = [[v_notArrow[i], u_notArrow[i]] for i in range(len(u_notArrow))]

    labels = np.array([1] * len(u_arrow) + [-1] * len(u_notArrow))
    trainingData = np.array(training_arrow + training_notArrow, dtype=np.float32)

    # Train the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)
    svm.save("output/svm")

    # svm = cv2.ml.SVM_load("output/svm")

    v_good = []
    u_good = []

    v_bad = []
    u_bad = []

    green = (0, 255, 0)
    blue = (255, 0, 0)
    for i in range(100, 200):
        for j in range(180):
            sampleMat = np.array([[i, j]], dtype=np.float32)
            response = svm.predict(sampleMat)[1]
            if response == 1:
                v_good.append(i)
                u_good.append(j)
            elif response == -1:
                v_bad.append(i)
                u_bad.append(j)

    plt.plot(v_arrow, u_arrow, 'gx')
    plt.plot(v_notArrow, u_notArrow, 'rx')
    plt.plot(v_good, u_good, 'go')
    plt.plot(v_bad, u_bad, 'ro')

    plt.xlabel('V')
    plt.ylabel('U')
    plt.axis([100, 200, 0, 180])

    plt.show()

    # img_resize = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    # img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

    # segment(v_good, u_good, img_yuv)

    cv2.waitKey(0)