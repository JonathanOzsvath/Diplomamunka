import os
import cv2
import math
import numpy as np
import globalVariable as gv


def discretise_YUV(name, img_YUV, umin=0, vmin=0, du=10, dv=10):
    directory = "output/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    Y = [-1, 0, 0.5, 1]

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


if __name__ == '__main__':
    du = gv.d
    dv = gv.d

    name = 'darts_with_arrow'
    path = '../images/' + name + '.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    discretise_YUV(name, img_YUV)

    cv2.waitKey(0)
