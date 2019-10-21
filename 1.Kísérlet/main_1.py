import cv2
import numpy as np
import matplotlib.pyplot as plt

numberOfKeypoint = 1000


def bruteForce(keyPoint, description, hMin):
    description_bitstring_array = uint8ArrayToBitstring(description)
    sorted_hamming_distance_array = makeAllHammingDistance(description_bitstring_array, description_bitstring_array)
    firstArray = [i[0] for i in sorted_hamming_distance_array]

    filtered_hamming_distance_array = list(filter(lambda m: m[1] > hMin, firstArray))

    filtered_indices = [i[0] for i in filtered_hamming_distance_array]
    # duplikáció szűrése
    filtered_indices = sorted(list(dict.fromkeys(filtered_indices)))
    kp2 = getDrawKeypoints(keyPoint, filtered_indices)

    print(len(kp2))

    drawKeypoints(img1, kp2, fileName=str(hMin) + "_" + str(numberOfKeypoint) + "_OwnBF")


def drawKeypoints(image, keyPoint, fileName, isShow=False, fx=1.0, fy=1.0, isGray=True):
    if isGray:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image1 = image.copy()
    image2 = image.copy()

    for i in keyPoint:
        image1 = cv2.circle(image1, (int(i.pt[0]), int(i.pt[1])), int(i.size/2), (0, 255, 0), thickness=1)
        image2 = cv2.circle(image2, (int(i.pt[0]), int(i.pt[1])), 3, (0, 255, 0), thickness=-1)

    cv2.imwrite("output/" + fileName + "_withSize.jpg", image1)
    cv2.imwrite("output/" + fileName + ".jpg", image2)

    if isShow:
        image = cv2.resize(image1, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(fileName, image1)


def getDrawKeypoints(kp, draw_indices):
    """ Vissza adja a kirajzolandó kulcspontokat az indexek alapján."""
    drawKeyPoints = []
    for i in draw_indices:
        drawKeyPoints.append(kp[i])

    return drawKeyPoints


def makeAllHammingDistance(description_bitstring_array1, description_bitstring_array2):
    """ Két bitstringeket tartalmazó tömbből az összes variáció szeint hamming távolságot számol.
        Minden elemnél távolság alapján rendez"""
    hamming_distance_array = []
    for i in range(0, len(description_bitstring_array1)):
        hamming_distances = []
        for j in range(0, len(description_bitstring_array2)):
            if i != j:
                hamming_distances.append(
                    (j, hammingDistance(description_bitstring_array1[i], description_bitstring_array2[j])))

        sorted_hamming_distance = sorted(hamming_distances, key=lambda hd: hd[1])
        hamming_distance_array.append(sorted_hamming_distance)

    return hamming_distance_array


def uint8ArrayToBitstring(description):
    """A kulcspontokpól jövő desc vektort alakítja string tömbbé"""
    description_bitstring = []
    for i in range(0, description.shape[0]):
        string_array = []
        for j in range(0, description.shape[1]):
            string_array.append(np.binary_repr(description[i][j]).zfill(8))
        description_bitstring.append(''.join(string_array))

    return description_bitstring


def hammingDistance(bitstring1, bitstring2):
    """Két bitstring hamming távolságát adja vissza"""
    counter = 0
    for i in range(0, len(bitstring1)):
        if bitstring1[i] != bitstring2[i]:
            counter += 1

    return counter


def opencvBruteForce(keyPoint, description, hMin):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(description, description, k=2)
    matches2 = matches.copy()
    for match in matches:
        match.pop(0)

    filtered_hamming_distance_array = list(filter(lambda m: m[0].distance > hMin, matches))

    filtered_indices = [i[0].queryIdx for i in filtered_hamming_distance_array]
    kp2 = getDrawKeypoints(keyPoint, filtered_indices)

    drawKeypoints(img1, kp2, fileName= str(hMin) + "_" + str(numberOfKeypoint) + "_openCVBF")

    return len(kp2)


if __name__ == '__main__':
    img = cv2.imread('../images/darts1_2.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hMinArray = list(range(0, 256))
    hMinArray = [55, 60, 65, 70]

    # Initiate ORB
    orb = cv2.ORB_create(numberOfKeypoint)
    kp, des = orb.detectAndCompute(img1, None)

    remainPointsNumber = []

    for hMin in hMinArray:
        drawKeypoints(img1, kp, fileName=str(hMin) + "_" + str(numberOfKeypoint) + "_ORB")

        bruteForce(kp, des, hMin)
        kpNumber = opencvBruteForce(kp, des, hMin)

        print((hMin, kpNumber))
        remainPointsNumber.append((hMin, kpNumber))

    plt.plot([i[0] for i in remainPointsNumber], [i[1] for i in remainPointsNumber])
    plt.title(str(numberOfKeypoint) + "db kulcspont esetén")
    plt.ylabel("Megmaradt ponotk száma")
    plt.xlabel("hMin")
    plt.savefig("output/" + str(numberOfKeypoint) + "_plot.png")
    plt.show()


    cv2.waitKey(0)
