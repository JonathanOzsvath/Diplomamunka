import numpy as np
import cv2
import os
import image_matcher as im
import image_matcher_eval as ime
import prefilter
import postfilter
import RANSAC
import dart_board
import time
import matplotlib.pyplot as plt

numberOfKeypoint = 1000
numberOfCirclePointPerSector = 3
minHamming_prefilter = 20
max_correct_radius = 5.0


def runVideo(img_ref, name_perspective, path_perspective, drawImage=True, showImage=False, saveImage=True):
    mask_ransac_array = []

    cap = cv2.VideoCapture(path_perspective)

    if saveImage:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/' + name_perspective + '.mp4', fourcc, 20.0, (720, 1280))

    refPoints, circlePoints = dart_board.generateDartBoardEdgePoints(numberOfCirclePointPerSector)

    orb = cv2.ORB_create(numberOfKeypoint)
    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
    kp_ref, des_ref = prefilter.prefilter(kp_ref, des_ref, min_hamming=minHamming_prefilter)

    counter = 0
    while cap.isOpened():
        start = time.time()

        ret, frame = cap.read()

        if ret:
            frame = np.rot90(frame, 3)
            img_perspective = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kp_perspective, des_perspective = orb.detectAndCompute(img_perspective, None)

            matches = im.openCVBF(kp_ref, des_ref, kp_perspective, des_perspective, crossCheck=False)
            matches = postfilter.ratioFilter(matches, maxRatio=0.8)
            matches = [m for m, n in matches]

            homography_ransac, mask_ransac = RANSAC.ransac(kp_ref, kp_perspective, matches, max_correct_radius=max_correct_radius, min_match_count=50)
            if len(mask_ransac) != 0:
                mask_ransac = [m[0] for m in mask_ransac]
                mask_ransac_array.append(len([m for m in mask_ransac if m == 1]))
            else:
                counter += 1
                mask_ransac_array.append(0)
                out.write(frame)
                continue

            click_point_ref = ime.LoadPoints(os.path.splitext(path_ref)[0] + '.click')
            homography_matrix_ref, _ = cv2.findHomography(np.array(refPoints), np.array(click_point_ref))

            circlePoints_ref = ime.Project(circlePoints, homography_matrix_ref)
            refPoints_ref = ime.Project(refPoints, homography_matrix_ref)

            ransac_circlePoints = ime.Project(circlePoints_ref, homography_ransac)
            ransac_refPoints = ime.Project(refPoints_ref, homography_ransac)

            if drawImage:
                # img = ime.drawPoints(frame, ransac_circlePoints, name_perspective, isGray=False)
                img = dart_board.drawDartBoard(frame, ransac_refPoints, ransac_circlePoints, numberOfCirclePointPerSector, showImage=showImage)

            if saveImage:
                out.write(img)

            end = time.time()
            runTime = ime.getRunTime(start, end)

            counter += 1
            print('{}: rt: {}'.format(counter, runTime))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    if saveImage:
        out.release()
    cv2.destroyAllWindows()

    return mask_ransac_array


def savePlot(mask_ransac_array, name_perspective):
    x = range(len(mask_ransac_array))
    y = mask_ransac_array

    plt.fill_between(x, y, color="skyblue", alpha=0.4)
    plt.plot(x, y, color="skyblue")

    plt.title('Number of inliers / Frame')
    plt.xlabel('Frame number')
    plt.ylabel('#Inlier')

    plt.savefig('output/InlierPerFrame_' + name_perspective + '.png', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    name_ref = "video_ref"
    path_ref = '../images/' + name_ref + '.jpg'

    img_ref = cv2.imread(path_ref)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    name_perspective = "video1"
    path_perspective = '../images/' + name_perspective + '.mp4'

    mask_ransac_array = runVideo(img_ref, name_perspective, path_perspective, drawImage=True, showImage=False, saveImage=True)

    savePlot(mask_ransac_array, name_perspective)
