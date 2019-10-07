import cv2
import numpy as np


def main():
    img = cv2.imread('../images/darts1.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate ORB
    orb = cv2.ORB_create(100)

    kp, des = orb.detectAndCompute(img1, None)

    img2 = cv2.drawKeypoints(img1, kp, None, flags=None, color=(0, 255, 0))

    img2 = cv2.resize(img2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    s = uint8_array_to_bitstring(des[0])

    cv2.imshow("ORB", img2)
    cv2.waitKey(0)


def uint8_array_to_bitstring(uint8_array):
    string_array = []
    for i in uint8_array:
        string_array.append(np.binary_repr(i).zfill(8))

    return ''.join(string_array)


def hamming_distance(bitstring1, bitstring2):
    counter = 0
    for i in range(0..len(bitstring1)):
        if bitstring1[i] == bitstring2[i]:
            counter += 1


if __name__ == '__main__':
    main()
