import cv2
import os
import numpy as np

numberOfFrame = 0

if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    name_video = "video1"
    path_perspective = '../images/' + name_video + '.mp4'

    cap = cv2.VideoCapture(path_perspective)

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret and i == numberOfFrame:
            frame = np.rot90(frame, 3)

            cv2.imwrite('../images/{}_{}_frame.jpg'.format(name_video, i), frame)
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
