import os
import numpy as np
import cv2

from utils.vis import image_show


def dev_detector(dm_image,
                 output_dir,
                 min_contour_area=16,
                 min_rect_size=(15, 15),
                 color_level=2,
                 contours_in_dilate=False,
                 dilate_limit_rate=0.1,
                 if_nms=True,
                 nms_threshold=0.9,
                 vis=False,
                 **kwargs):

    # lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD,
    #                                     _scale=1,
    #                                     _sigma_scale=0.6)
    # lines = lsd.detect(dm_image.gray)


    os.makedirs(os.path.join(output_dir, "detect"), exist_ok=True)
    image, gradient = preprocess(dm_image.gray, color_level, vis)
    convex, contours = convex_dilate(image, min_contour_area, contours_in_dilate, dilate_limit_rate)
    # contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target, rect_candidate = parse_contours(contours, min_rect_size, dm_image.img)

    # vis
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    convex = cv2.cvtColor(convex, cv2.COLOR_GRAY2BGR)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(output_dir, "detect", os.path.basename(dm_image.img_path)),
                np.hstack((gradient, image, convex, target)))
    if vis:
        image_show(np.hstack((gradient, image, convex, target)))
    return rect_candidate


def nms():
    pass


def parse_contours(contours, min_rect_size, src):
    target = src.copy()
    rect_candidate = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rect_contour = np.int0(cv2.boxPoints(rect))
        if rect[1][0] > min_rect_size[0] and rect[1][1] > min_rect_size[1]:
            rect_candidate.append(rect_contour)
            cv2.drawContours(target, [rect_contour], 0, (255, 255, 0), 1)
    return target, rect_candidate


def preprocess(src, color_level, vis):
    target = src.copy()
    kernel_3x3 = np.ones((3, 3), np.uint8)
    # TODO: adaptive BilateralFilter
    smoothed = cv2.bilateralFilter(src=target, d=15, sigmaColor=40, sigmaSpace=10)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_3x3, iterations=1)
    gradient_w = cv2.Sobel(smoothed, cv2.CV_16S, dx=1, dy=0, ksize=3)
    gradient_h = cv2.Sobel(smoothed, cv2.CV_16S, dx=0, dy=1, ksize=3)
    gradient = cv2.convertScaleAbs(cv2.addWeighted(gradient_w, 0.5, gradient_h, 0.5, 0))
    # gradient = cv2.convertScaleAbs(np.sqrt(gradient_w ** 2 + gradient_h ** 2).astype(np.uint8))
    gradient_cluster, _, centers = color_cluster(gradient, color_level=color_level,
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
    return cv2.split(gradient_cluster)[target_channel], gradient


def convex_dilate(src, min_contour_area, contours_in_dilate, dilate_limit_rate):
    target = src.copy()
    image_show(target)
    target = cv2.morphologyEx(target, cv2.MORPH_TOPHAT , cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    image_show(target)
    previous_contours_num = -1
    previous_contours = []
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
            hull = cv2.convexHull(contour, clockwise=True)
            if contours_in_dilate:
                previous_contours.append(hull)
            if cv2.contourArea(hull) * dilate_limit_rate < cv2.contourArea(contour):
                cv2.drawContours(target, [hull], -1, 255, -1)
            else:
                cv2.drawContours(target, [contour], -1, 255, -1)
    if contours_in_dilate:
        contours = previous_contours
    return target, contours


def color_cluster(src, color_level, color_map=None):
    channel = 1 if len(src.shape) == 2 else src.shape[2]
    pixel_list = np.float32(src.reshape((-1, channel)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_list, color_level, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    if color_map:
        colors = np.uint8(color_map)
    else:
        colors = np.uint8(centers)
    w, h = src.shape[0: 2]
    image = colors[labels.flatten()].reshape((w, h, 3))
    return image, labels.reshape((w, h)), centers
