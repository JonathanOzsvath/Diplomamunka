import cv2
import numpy as np

number_of_keypoint = 500
h_min = 110


def main():
    img = cv2.imread('../images/darts1_2.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate ORB
    orb = cv2.ORB_create(number_of_keypoint)

    kp, des = orb.detectAndCompute(img1, None)

    description_bitstring_array = uint8_array_to_bitstring(des)
    sorted_hamming_distance_array = make_all_hamming_distance(description_bitstring_array)
    filtered_hamming_distance_array = filter_by_hamming_distance(sorted_hamming_distance_array)

    filtered_indices = get_draw_indices(filtered_hamming_distance_array)
    kp2 = get_draw_keypoints(kp, filtered_indices)

    draw_keypoints(img1, kp, 0.25, 0.25, "ORB")
    draw_keypoints(img1, kp2, 0.25, 0.25, "Filtered")

    print(len(kp2))

    cv2.waitKey(0)


def draw_keypoints(img, kp, fx=1.0, fy=1.0, title='', isGray=True):
    if isGray:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in kp:
        img = cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 10, (0, 255, 0), thickness=-1)

    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(title, img)


def get_draw_keypoints(kp, draw_indices):
    draw_keypoints = []
    for i in draw_indices:
        draw_keypoints.append(kp[i])

    return draw_keypoints


def get_draw_indices(filtered_hamming_distance_array):
    indices = list(range(0, number_of_keypoint))
    copy_filtered_hamming_distance_array = filtered_hamming_distance_array.copy()

    for element in copy_filtered_hamming_distance_array:
        for e in element:
            if e[0] in indices:
                indices.remove(e[0])
                copy_filtered_hamming_distance_array.remove(filtered_hamming_distance_array[e[0]])

    return indices


def filter_by_hamming_distance(sorted_hamming_distance_array):
    filtered_hamming_distance_array = []
    for element in sorted_hamming_distance_array:
        filtered_hamming_distance_array.append(list(filter(lambda h: h[1] < h_min, element)))

    return filtered_hamming_distance_array


def make_all_hamming_distance(description_bitstring_array):
    hamming_distance_array = []
    for i in range(0, len(description_bitstring_array)):
        hamming_distances = []
        for j in range(0, len(description_bitstring_array)):
            if i != j:
                hamming_distances.append(
                    (j, hamming_distance(description_bitstring_array[i], description_bitstring_array[j])))

        sorted_hamming_distance = sorted(hamming_distances, key=lambda hd: hd[1])
        hamming_distance_array.append(sorted_hamming_distance)

    return hamming_distance_array


def uint8_array_to_bitstring(description):
    description_bitstring = []
    for i in range(0, description.shape[0]):
        string_array = []
        for j in range(0, description.shape[1]):
            string_array.append(np.binary_repr(description[i][j]).zfill(8))
        description_bitstring.append(''.join(string_array))

    return description_bitstring


def hamming_distance(bitstring1, bitstring2):
    counter = 0
    for i in range(0, len(bitstring1)):
        if bitstring1[i] == bitstring2[i]:
            counter += 1

    return counter


if __name__ == '__main__':
    main()
