import os
import numpy as np
import cv2
import math

from utils.vis import image_show


def dev_detector(dm_image, output_dir, min_contour_area=16, vis=True, **kwargs):
    image = preprocess(dm_image.gray, vis)
    result = convex_dilate(image, min_contour_area, 0.1)

    if vis:
        image_show(np.hstack((image, result)))
    # l_shape_finder(image, min_contour_area, vis)
    pass


def preprocess(src, vis):
    target = src.copy()
    kernel_3x3 = np.ones((3, 3), np.uint8)
    smoothed = cv2.bilateralFilter(src=target, d=15, sigmaColor=40, sigmaSpace=10)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_3x3, iterations=1)
    gradient_w = cv2.Sobel(smoothed, cv2.CV_16S, dx=1, dy=0, ksize=3)
    gradient_h = cv2.Sobel(smoothed, cv2.CV_16S, dx=0, dy=1, ksize=3)
    gradient = cv2.convertScaleAbs(cv2.addWeighted(gradient_w, 0.5, gradient_h, 0.5, 0))
    # gradient = cv2.convertScaleAbs(np.sqrt(gradient_w ** 2 + gradient_h ** 2).astype(np.uint8))
    gradient_cluster, _, centers = color_cluster(gradient, color_level=3,
                                                 color_map=[[255, 0, 0], [0, 255, 0], [0, 0, 255],
                                                            [255, 255, 0], [0, 255, 255], [255, 0, 255],
                                                            [0, 0, 0], [255, 255, 255]])
    # TODO: optimized for JHT, shouldn't target at only one channel.
    target_channel = centers.argmax()
    # gradient_erode = cv2.erode(gradient, kernel_3x3, iterations=1)
    if vis:
        image_show(np.vstack((
            np.hstack((src, smoothed, gradient)),
            np.hstack((cv2.split(gradient_cluster)[0], cv2.split(gradient_cluster)[1], cv2.split(gradient_cluster)[2]))
        )))
    return cv2.split(gradient_cluster)[target_channel]


def convex_dilate(src, min_contour_area, dilate_limit_rate):
    target = src.copy()
    previous_contours_num = -1
    while True:
        contours, hierarchy = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if previous_contours_num == -1:
            target = np.zeros(src.shape).astype(np.uint8)
            contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
        if len(contours) != previous_contours_num:
            previous_contours_num = len(contours)
        else:
            break
        for contour in contours:
            length = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, length * 0.01, True)
            hull = cv2.convexHull(approx, clockwise=True)
            if cv2.contourArea(hull) * dilate_limit_rate < cv2.contourArea(contour):
                cv2.drawContours(target, [hull], -1, 255, -1)
            else:
                cv2.drawContours(target, [contour], -1, 255, -1)
    return target


def color_cluster(src, color_level, color_map=None):
    channel = 1 if len(src.shape) == 2 else src.shape[2]
    pixel_list = np.float32(src.reshape((-1, channel)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(pixel_list, color_level, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    if color_map:
        colors = np.uint8(color_map)
    else:
        colors = np.uint8(centers)
    w, h = src.shape[0: 2]
    image = colors[labels.flatten()].reshape((w, h, 3))
    return image, labels.reshape((w, h)), centers
