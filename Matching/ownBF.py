import numpy as np


def bruteForce(kpReference, desReference, kpPerspective, desPerspective):
    des_reference_bitstring_array = uint8ArrayToBitstring(desReference)
    des_perspective_bitstring_array = uint8ArrayToBitstring(desPerspective)
    sorted_hamming_distance_array = makeAllHammingDistance(des_reference_bitstring_array, des_perspective_bitstring_array)
    firstTwo = [(i[0], i[1]) for i in sorted_hamming_distance_array]

    return firstTwo


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