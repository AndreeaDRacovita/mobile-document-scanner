# OpenCV Tutorial from Murtaza's Workshop - Robotics and AI

import numpy as np
import cv2

width = 640
height = 640
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, width)
cap.set(4, height)
cap.set(10, 150)


def get_contours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 5)
    return biggest


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(axis=1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def get_warp(img, biggest):
    if biggest.size == 0:
        return img
        pass
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width, height))

    img_cropped = img_output[10:img_output.shape[0]-10, 10:img_output.shape[1]-10]
    img_cropped = cv2.resize(img_cropped, (width, height))
    return img_output
    # return img_cropped


def pre_processing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dilation = cv2.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv2.erode(img_dilation, kernel, iterations=1)
    return img_threshold


while True:
    success, img = cap.read()
    img = cv2.flip(img, -1)
    img = cv2.resize(img, (width, height))
    img_contour = img.copy()
    img_thres = pre_processing(img)
    biggest = get_contours(img_thres)
    img_warped = get_warp(img, biggest)

    cv2.imshow("Result", img_warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
