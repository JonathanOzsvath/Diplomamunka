import os
import cv2
import math
import numpy as np
import globalVariable as gv


def discretise_YUV(name, img_YUV, umin=0, vmin=0, du=10, dv=10):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    Y = [-1, 0, 127, 255]

    height, width = img_YUV.shape[:2]

    for y_value in Y:
        img_YUV_discretise = np.zeros(img_YUV.shape, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if y_value == -1:
                    y, u, v = img_YUV[i, j]
                else:
                    u, v = img_YUV[i, j][1:]
                    y = y_value

                color_u = du * (u // du) + (du / 2)
                color_v = dv * (v // dv) + (dv / 2)

                img_YUV_discretise[i, j] = [y, color_u, color_v]

        img_YUV2RGB_discretise = cv2.cvtColor(img_YUV_discretise, cv2.COLOR_YUV2BGR)
        if y_value == -1:
            cv2.imwrite(directory + name + "_discretise.jpg", img_YUV2RGB_discretise)
        else:
            cv2.imwrite(directory + name + "_discretise_" + str(y_value) + ".jpg", img_YUV2RGB_discretise)


def maskAverage(name, img, mask):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    height, width = img.shape[:2]
    arrow = []

    for y in range(height):
        for x in range(width):
            if mask[y, x] > 200:
                arrow.append(img[y, x])

    np_arrow = np.array(arrow, dtype=np.uint8)
    mean = [np.uint8(np.mean(np_arrow[:, 0])), np.uint8(np.mean(np_arrow[:, 1])), np.uint8(np.mean(np_arrow[:, 2]))]

    img_mean_arrow = np.full((500, 500, 3), mean)

    img_mean_arrow_YUV = cv2.cvtColor(img_mean_arrow, cv2.COLOR_BGR2YUV)
    img_mean_arrow_YUV_0 = img_mean_arrow_YUV.copy()
    img_mean_arrow_YUV_128 = img_mean_arrow_YUV.copy()
    img_mean_arrow_YUV_255 = img_mean_arrow_YUV.copy()

    img_mean_arrow_YUV_0[:, :, 0] = 0
    img_mean_arrow_YUV_0 = cv2.cvtColor(img_mean_arrow_YUV_0, cv2.COLOR_YUV2BGR)
    img_mean_arrow_YUV_128[:, :, 0] = 128
    img_mean_arrow_YUV_128 = cv2.cvtColor(img_mean_arrow_YUV_128, cv2.COLOR_YUV2BGR)
    img_mean_arrow_YUV_255[:, :, 0] = 255
    img_mean_arrow_YUV_255 = cv2.cvtColor(img_mean_arrow_YUV_255, cv2.COLOR_YUV2BGR)

    cv2.imwrite(directory + name + "_mean.jpg", img_mean_arrow)
    cv2.imwrite(directory + name + "_mean_YUV_Y0.jpg", img_mean_arrow_YUV_0)
    cv2.imwrite(directory + name + "_mean_YUV_Y128.jpg", img_mean_arrow_YUV_128)
    cv2.imwrite(directory + name + "_mean_YUV_Y255.jpg", img_mean_arrow_YUV_255)

if __name__ == '__main__':
    du = gv.d
    dv = gv.d

    name = 'darts_with_arrow'
    path = '../images/' + name + '.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    name_mask = name + '_mask_positive'
    path_mask = '../images/' + name_mask + '.jpg'
    img_mask = cv2.imread(path_mask, 0)

    # discretise_YUV(name, img_YUV)
    maskAverage(name, img, img_mask)

    cv2.waitKey(0)
