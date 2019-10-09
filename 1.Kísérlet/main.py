import cv2
import numpy as np

numberOfKeypoint = 500
hMin = 110


def bruteForce(keyPoint, description):
    description_bitstring_array = uint8ArrayToBitstring(description)
    sorted_hamming_distance_array = makeAllHammingDistance(description_bitstring_array)
    filtered_hamming_distance_array = filterByHammingDistance(sorted_hamming_distance_array)

    filtered_indices = getDrawIndices(filtered_hamming_distance_array)
    kp2 = getDrawKeypoints(keyPoint, filtered_indices)

    drawKeypoints(img1, kp2, 0.25, 0.25, "Filtered")

    cv2.waitKey(0)


def drawKeypoints(image, keyPoint, fx=1.0, fy=1.0, title='', isGray=True):
    if isGray:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in keyPoint:
        image = cv2.circle(image, (int(i.pt[0]), int(i.pt[1])), 10, (0, 255, 0), thickness=-1)

    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(title, image)


def getDrawKeypoints(kp, draw_indices):
    drawKeyPoints = []
    for i in draw_indices:
        drawKeyPoints.append(kp[i])

    return drawKeyPoints


def getDrawIndices(filtered_hamming_distance_array):
    indices = list(range(0, numberOfKeypoint))
    copy_filtered_hamming_distance_array = filtered_hamming_distance_array.copy()

    for element in copy_filtered_hamming_distance_array:
        for e in element:
            if e[0] in indices:
                indices.remove(e[0])
                copy_filtered_hamming_distance_array.remove(filtered_hamming_distance_array[e[0]])

    return indices


def filterByHammingDistance(sorted_hamming_distance_array):
    filtered_hamming_distance_array = []
    for element in sorted_hamming_distance_array:
        filtered_hamming_distance_array.append(list(filter(lambda h: h[1] < hMin, element)))

    return filtered_hamming_distance_array


def makeAllHammingDistance(description_bitstring_array):
    hamming_distance_array = []
    for i in range(0, len(description_bitstring_array)):
        hamming_distances = []
        for j in range(0, len(description_bitstring_array)):
            if i != j:
                hamming_distances.append(
                    (j, hammingDistance(description_bitstring_array[i], description_bitstring_array[j])))

        sorted_hamming_distance = sorted(hamming_distances, key=lambda hd: hd[1])
        hamming_distance_array.append(sorted_hamming_distance)

    return hamming_distance_array


def uint8ArrayToBitstring(description):
    description_bitstring = []
    for i in range(0, description.shape[0]):
        string_array = []
        for j in range(0, description.shape[1]):
            string_array.append(np.binary_repr(description[i][j]).zfill(8))
        description_bitstring.append(''.join(string_array))

    return description_bitstring


def hammingDistance(bitstring1, bitstring2):
    counter = 0
    for i in range(0, len(bitstring1)):
        if bitstring1[i] == bitstring2[i]:
            counter += 1

    return counter


def opencvBruteForce(description):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(description, description, k=2)
    matches2 = matches.copy()
    for match in matches:
        match.pop(0)

    for match in matches:
        if match[0].distance < 50:
            matches2.remove(match)
    a = 0


if __name__ == '__main__':
    img = cv2.imread('../images/darts1_2.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate ORB
    orb = cv2.ORB_create(numberOfKeypoint)
    kp, des = orb.detectAndCompute(img1, None)
    # drawKeypoints(img1, kp, 0.25, 0.25, "ORB")

    # brute_force(kp, des)
    opencvBruteForce(des)
