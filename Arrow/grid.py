import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import ArrowSets


def computeColorHist2D(arrows, notArrows, umin, umax, vmin, vmax, du=5, dv=5):
    color_histogram_positive = np.zeros((int(255 / du), int(255 / dv)), dtype=np.int)
    color_histogram_negative = np.zeros((int(255 / du), int(255 / dv)), dtype=np.int)

    for u, v in arrows:
        i = math.floor((u - umin) / du)
        j = math.floor((v - vmin) / dv)
        color_histogram_positive[i, j] = color_histogram_positive[i, j] + 1

    for u, v in notArrows:
        i = math.floor((u - umin) / du)
        j = math.floor((v - vmin) / dv)
        color_histogram_negative[i, j] = color_histogram_negative[i, j] + 1

    np.savetxt('output/color_histogram_positive.txt', color_histogram_positive, fmt='%1i')
    np.savetxt('output/color_histogram_negative.txt', color_histogram_negative, fmt='%1i')

    print('Arrows: {}'.format(len(arrows)))
    print('Not Arrows: {}'.format(len(notArrows)))
    return color_histogram_positive, color_histogram_negative


def makePlot(arrows, notArrows, umin=0, umax=255, vmin=0, vmax=255, du=5, dv=5):
    arrows_u = [i[0] for i in arrows]
    arrows_v = [i[1] for i in arrows]

    notArrows_u = [i[0] for i in notArrows]
    notArrows_v = [i[1] for i in notArrows]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    major_ticks = np.arange(0, 256, 20)
    minor_ticks = np.arange(0, 256, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.plot(notArrows_u, notArrows_v, 'rx')
    ax.plot(arrows_u, arrows_v, 'gx')
    ax.grid(which='both')
    ax.grid(which='major', alpha=1.0)

    plt.xlabel('U')
    plt.ylabel('V')

    plt.axis([0, 256, 0, 256])

    plt.title('UV plot')
    plt.savefig('output/uv_plot.png', bbox_inches="tight")
    plt.show()


def computePMatrix(color_histogram_positive, color_histogram_negative):
    width, height = color_histogram_positive.shape
    P_positive = np.zeros(color_histogram_positive.shape)

    sum = np.add(color_histogram_positive, color_histogram_negative)

    for i in range(width):
        for j in range(height):
            if sum[i, j] != 0:
                P_positive[i, j] = color_histogram_positive[i, j] / sum[i, j]
            else:
                P_positive[i, j] = 0

    return P_positive


def computeBinaryMatrix(color_histogram_positive, color_histogram_negative):
    width, height = color_histogram_positive.shape
    binaryMatrix = np.zeros(color_histogram_positive.shape)

    for i in range(width):
        for j in range(height):
            if color_histogram_positive[i, j] > color_histogram_negative[i, j]:
                binaryMatrix[i, j] = 1

    return binaryMatrix


def binarySegmentation(img_YUV, binaryMatrix, umin, umax, vmin, vmax, du=5, dv=5):
    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((u - umin) / du)
            j = math.floor((v - vmin) / dv)

            if binaryMatrix[i, j] == 1:
                img[y, x] = img_YUV[y, x]

    cv2.imshow('binarySegmentation', img)


def probabilitySegmentation(name, img_YUV, P_positive, umin, umax, vmin, vmax, du=5, dv=5):
    directory = "output/probability/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((u - umin) / du)
            j = math.floor((v - vmin) / dv)

            if P_positive[i, j] >= 0.2:
                img[y, x] = img_YUV[y, x]

    cv2.imshow('probabilitySegmentation', img)
    cv2.imwrite(directory + name + '_segmentation.jpg', img)


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    arrows, notArrows = ArrowSets.loadArrowSets()

    # makePlot(arrows, notArrows)

    # color_histogram_positive, color_histogram_negative = computeColorHist2D(arrows, notArrows, 0, 255, 0, 255)

    color_histogram_positive = np.loadtxt('output/color_histogram_positive.txt', dtype=np.int)
    color_histogram_negative = np.loadtxt('output/color_histogram_negative.txt', dtype=np.int)

    # positive_max = np.ndarray.max(color_histogram_positive)
    # negative_max = np.ndarray.max(color_histogram_negative)
    #
    # img_color_histogram_positive = np.array(color_histogram_positive / positive_max * 255, dtype=np.uint8)
    # img_color_histogram_positive = cv2.resize(img_color_histogram_positive, None, fx=20, fy=20, interpolation=cv2.INTER_CUBIC)
    # img_color_histogram_positive = cv2.applyColorMap(img_color_histogram_positive, cv2.COLORMAP_JET)
    # cv2.imshow('img_color_histogram_positive', img_color_histogram_positive)
    # cv2.imwrite('output/img_color_histogram_positive.jpg', img_color_histogram_positive)
    #
    # img_color_histogram_negative = np.array(color_histogram_negative / negative_max * 255, dtype=np.uint8)
    # img_color_histogram_negative = cv2.resize(img_color_histogram_negative, None, fx=20, fy=20, interpolation=cv2.INTER_CUBIC)
    # img_color_histogram_negative = cv2.applyColorMap(img_color_histogram_negative, cv2.COLORMAP_JET)
    # cv2.imshow('img_color_histogram_negative', img_color_histogram_negative)
    # cv2.imwrite('output/img_color_histogram_negative.jpg', img_color_histogram_negative)

    P_positive = computePMatrix(color_histogram_positive, color_histogram_negative)
    img_P_positive = np.array(P_positive * 255, dtype=np.uint8)
    img_P_positive = cv2.resize(img_P_positive, None, fx=20, fy=20, interpolation=cv2.INTER_CUBIC)
    img_P_positive = cv2.applyColorMap(img_P_positive, cv2.COLORMAP_JET)
    cv2.imshow('img_P_positive', img_P_positive)
    cv2.imwrite('output/img_P_positive.jpg', img_P_positive)

    # binaryMatrix = computeBinaryMatrix(color_histogram_positive, color_histogram_negative)
    # binaryMatrix = np.array(binaryMatrix * 255, dtype=np.uint8)
    # binaryMatrix = cv2.resize(binaryMatrix, None, fx=20, fy=20, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('binaryMatrix', binaryMatrix)
    # cv2.imwrite('output/binaryMatrix.jpg', binaryMatrix)

    # names = ['darts_with_arrow', 'darts_with_arrow2', 'darts_with_arrow3', 'darts_with_arrow4', 'darts_with_arrow5', 'darts_with_arrow6',
    #          'darts_with_arrow7', 'darts_with_arrow8', 'darts_with_arrow9', 'darts_with_arrow10']

    names = ['darts_with_arrow11', 'darts_with_arrow12']

    for name in names:
        path = '../images/' + name + '.jpg'
        img = cv2.imread(path)
        img_resize = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

        # binarySegmentation(img_yuv, binaryMatrix, 0, 255, 0, 255)
        probabilitySegmentation(name, img_yuv, P_positive, 0, 255, 0, 255)

    cv2.waitKey(0)
