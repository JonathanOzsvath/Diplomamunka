import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import ArrowSets

resize = 40
du = 10
dv = 10


def computeColorHist2D(arrows, notArrows, umin, umax, vmin, vmax, du=10, dv=10):
    color_histogram_positive = np.zeros((int(255 / du), int(255 / dv)), dtype=np.int)
    color_histogram_negative = np.zeros((int(255 / du), int(255 / dv)), dtype=np.int)

    for u, v in arrows:
        i = math.floor((v - vmin) / dv)
        j = math.floor((u - umin) / du)
        color_histogram_positive[i, j] = color_histogram_positive[i, j] + 1

    for u, v in notArrows:
        i = math.floor((v - vmin) / dv)
        j = math.floor((u - umin) / du)
        color_histogram_negative[i, j] = color_histogram_negative[i, j] + 1

    np.savetxt('output/color_histogram_positive.txt', color_histogram_positive, fmt='%1i')
    np.savetxt('output/color_histogram_negative.txt', color_histogram_negative, fmt='%1i')

    print('Arrows: {}'.format(len(arrows)))
    print('Not Arrows: {}'.format(len(notArrows)))
    return color_histogram_positive, color_histogram_negative


def makePlot(arrows, notArrows):
    arrows_u = [i[0] for i in arrows]
    arrows_v = [i[1] for i in arrows]

    notArrows_u = [i[0] for i in notArrows]
    notArrows_v = [i[1] for i in notArrows]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    major_ticks = np.arange(0, 256, 20)
    minor_ticks = np.arange(0, 256, du)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.plot(notArrows_u, notArrows_v, 'rx')
    ax.plot(arrows_u, arrows_v, 'gx')
    ax.plot(arrows_poly_u, arrows_poly_v, 'b-')
    ax.plot(notArrows_poly_u, notArrows_poly_v, 'k-')

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


def computeRGBMatrix(arrows, notArrows, umin, umax, vmin, vmax, du=10, dv=10):
    sumMatrix = np.zeros((int(255 / du), int(255 / dv), 3), dtype=np.int)
    yuvMatrix = np.zeros((int(255 / du), int(255 / dv), 3), dtype=np.uint8)
    width, height = sumMatrix.shape[:2]

    for u, v in arrows:
        i = math.floor((v - vmin) / dv)
        j = math.floor((u - umin) / du)
        sumMatrix[i, j] = sumMatrix[i, j] + [u, v, 1]

    for u, v in notArrows:
        i = math.floor((v - vmin) / dv)
        j = math.floor((u - umin) / du)
        sumMatrix[i, j] = sumMatrix[i, j] + [u, v, 1]

    for i in range(width):
        for j in range(height):
            if sumMatrix[i, j][2] == 0:
                # sumMatrix[i, j][2] = 1
                yuvMatrix[i, j] = [0, 128, 128]
            else:
                yuvMatrix[i, j] = [0, sumMatrix[i, j][0] / sumMatrix[i, j][2], sumMatrix[i, j][1] / sumMatrix[i, j][2]]

    img_rgbMatrix = cv2.cvtColor(yuvMatrix, cv2.COLOR_YUV2BGR)

    return img_rgbMatrix


def binarySegmentation(img_YUV, binaryMatrix, umin, umax, vmin, vmax, du=10, dv=10):
    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((v - vmin) / dv)
            j = math.floor((u - umin) / du)

            if binaryMatrix[i, j] == 1:
                img[y, x] = img_YUV[y, x]

    cv2.imshow('binarySegmentation', img)


def probabilitySegmentation(name, img_YUV, P_positive, umin, umax, vmin, vmax, du=10, dv=10):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape[:2], dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((v - vmin) / dv)
            j = math.floor((u - umin) / du)

            img[y, x] = int(P_positive[i, j] * 255)

    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # cv2.imshow('probabilitySegmentation', img)
    cv2.imwrite(directory + name + '_probability.jpg', img)


def probabilitySegmentation2(name, img_YUV, P_positive, umin, umax, vmin, vmax, du=10, dv=10):
    directory = "output/name"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((v - vmin) / dv)
            j = math.floor((u - umin) / du)

            if P_positive[i, j] >= 0.1:
                img[y, x] = img_YUV[y, x]

    cv2.imshow('probabilitySegmentation', img)
    cv2.imwrite(directory + name + '_segmentation.jpg', img)


def colorDiscreteSegmentation(name, img_YUV, rgbMatrix, umin, umax, vmin, vmax, du=10, dv=10):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img_YUV.shape[:2]
    img = np.zeros(img_YUV.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            u, v = img_YUV[y, x][1:]
            i = math.floor((v - vmin) / dv)
            j = math.floor((u - umin) / du)

            img[y, x] = rgbMatrix[i, j]

    # cv2.imshow(name, img)
    cv2.imwrite(directory + name + "_discretise.jpg", img)


def drawPoly(img, arrows_poly, notArrows_poly):
    img = cv2.polylines(img, np.int32([arrows_poly]), True, (0, 0, 0), thickness=3)
    img = cv2.polylines(img, np.int32([notArrows_poly]), True, (255, 255, 255), thickness=3)

    return img


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    arrows, notArrows = ArrowSets.loadArrowSets()

    arrows_poly_u = [130, 160, 150, 130, 100, 110, 130]
    arrows_poly_v = [170, 115, 120, 105, 150, 165, 170]
    arrows_poly = np.array([[int(arrows_poly_u[i] / du * resize), int(arrows_poly_v[i] / dv * resize)] for i in range(len(arrows_poly_u) - 1)])

    notArrows_poly_u = [125, 150, 155, 225, 125, 110, 75, 105, 100, 125]
    notArrows_poly_v = [250, 230, 165, 95, 40, 125, 160, 160, 225, 250]
    notArrows_poly = np.array([[int(notArrows_poly_u[i] / du * resize), int(notArrows_poly_v[i] / dv * resize)] for i in range(len(notArrows_poly_u) - 1)])

    makePlot(arrows, notArrows)

    color_histogram_positive, color_histogram_negative = computeColorHist2D(arrows, notArrows, 0, 255, 0, 255)

    # color_histogram_positive = np.loadtxt('output/color_histogram_positive.txt', dtype=np.int)
    # color_histogram_negative = np.loadtxt('output/color_histogram_negative.txt', dtype=np.int)

    # positive_max = np.ndarray.max(color_histogram_positive)
    # negative_max = np.ndarray.max(color_histogram_negative)

    positive_max = 100
    negative_max = 5000

    img_color_histogram_positive = np.array(color_histogram_positive / positive_max * 255, dtype=np.uint8)
    img_color_histogram_positive = cv2.resize(img_color_histogram_positive, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    img_color_histogram_positive = cv2.applyColorMap(img_color_histogram_positive, cv2.COLORMAP_JET)
    img_color_histogram_positive = drawPoly(img_color_histogram_positive, arrows_poly, notArrows_poly)
    img_color_histogram_positive = cv2.flip(img_color_histogram_positive, 0)
    cv2.imshow('img_color_histogram_positive', img_color_histogram_positive)
    cv2.imwrite('output/img_color_histogram_positive.jpg', img_color_histogram_positive)

    img_color_histogram_negative = np.array(color_histogram_negative / negative_max * 255, dtype=np.uint8)
    img_color_histogram_negative = cv2.resize(img_color_histogram_negative, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    img_color_histogram_negative = cv2.applyColorMap(img_color_histogram_negative, cv2.COLORMAP_JET)

    img_color_histogram_negative = drawPoly(img_color_histogram_negative, arrows_poly, notArrows_poly)

    img_color_histogram_negative = cv2.flip(img_color_histogram_negative, 0)
    cv2.imshow('img_color_histogram_negative', img_color_histogram_negative)
    cv2.imwrite('output/img_color_histogram_negative.jpg', img_color_histogram_negative)

    P_positive = computePMatrix(color_histogram_positive, color_histogram_negative)
    img_P_positive = np.array(P_positive * 255, dtype=np.uint8)
    img_P_positive = cv2.resize(img_P_positive, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    img_P_positive = cv2.applyColorMap(img_P_positive, cv2.COLORMAP_JET)
    img_P_positive = drawPoly(img_P_positive, arrows_poly, notArrows_poly)
    img_P_positive = cv2.flip(img_P_positive, 0)
    cv2.imshow('img_P_positive', img_P_positive)
    cv2.imwrite('output/img_P_positive.jpg', img_P_positive)

    # binaryMatrix = computeBinaryMatrix(color_histogram_positive, color_histogram_negative)
    # img_binaryMatrix = np.array(binaryMatrix * 255, dtype=np.uint8)
    # img_binaryMatrix = cv2.resize(img_binaryMatrix, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    # img_binaryMatrix = cv2.flip(img_binaryMatrix, 0)
    # cv2.imshow('binaryMatrix', img_binaryMatrix)
    # cv2.imwrite('output/binaryMatrix.jpg', img_binaryMatrix)

    rgbMatrix = computeRGBMatrix(arrows, notArrows, 0, 255, 0, 255)
    img_rgbMatrix = cv2.resize(rgbMatrix, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    img_rgbMatrix = drawPoly(img_rgbMatrix, arrows_poly, notArrows_poly)
    img_rgbMatrix = cv2.flip(img_rgbMatrix, 0)
    cv2.imshow('rgbMatrix', img_rgbMatrix)
    cv2.imwrite('output/rgbMatrix.jpg', img_rgbMatrix)

    names = ['darts_with_arrow', 'darts_with_arrow2', 'darts_with_arrow3', 'darts_with_arrow4', 'darts_with_arrow5', 'darts_with_arrow6',
             'darts_with_arrow7', 'darts_with_arrow8', 'darts_with_arrow9', 'darts_with_arrow10']

    # names = ['darts_with_arrow']

    for name in names:
        path = '../images/' + name + '.jpg'
        img = cv2.imread(path)
        img_resize = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YUV)

        # binarySegmentation(img_yuv, binaryMatrix, 0, 255, 0, 255)
        probabilitySegmentation(name, img_yuv, P_positive, 0, 255, 0, 255)
        colorDiscreteSegmentation(name, img_yuv, rgbMatrix, 0, 255, 0, 255)

    cv2.waitKey(0)
