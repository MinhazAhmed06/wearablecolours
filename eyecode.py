import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def eye_color(image):
    color_dict = {}
    hue_list = []

    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))

    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/11

    cv2.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    # cv2.imshow('Eyes Detected', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for y in range(h):
        for x in range(w):
            if imgMask[y, x] != 0:
                color = tuple(map(int, img[y, x]))
                hue_list.append(img[y,x][0])
                if color in color_dict:
                    color_dict[color] += 1
                else:
                    color_dict[color] = 1

    avg_hue = sum(hue_list) / len(hue_list)
    return avg_hue

image = cv2.imread('image.jpg')
print(eye_color(image))