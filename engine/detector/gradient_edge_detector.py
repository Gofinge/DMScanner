import os
import numpy as np
import cv2
from utils.miscellaneous import get_center, calc_euclidean_distance, cal_iou
from utils.augmentation import normalize
from utils.vis import image_show


def dm_gradient_edge_detector(dm_image, output_dir, **kwargs):
    w_sobel_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    h_sobel_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    morph_kernel_3x3 = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])
    morph_kernel_5x5 = np.array([[1, 1, 1, 1, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1]])
    filter_size = (25, 25)
    block_size = 35
    thread_offset = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)

    results = []
    image_raw = cv2.cvtColor(dm_image.gray, cv2.COLOR_GRAY2BGR).copy()

    for level in range(len(dm_image.gray_pyramid)):
        image = dm_image.gray_pyramid[level]
        if level == 0:
            image_show(image, "raw")
        image_smoothed = cv2.bilateralFilter(image, d=13, sigmaColor=46, sigmaSpace=8)
        # gradient
        gradient_w = cv2.filter2D(image_smoothed, cv2.CV_32F, w_sobel_kernel)
        gradient_h = cv2.filter2D(image_smoothed, cv2.CV_32F, h_sobel_kernel)
        gradient_scale_o = np.sqrt(gradient_w ** 2 + gradient_h ** 2)
        gradient_scale_o[gradient_scale_o > 255] = 255
        gradient_scale_o = normalize(gradient_scale_o)
        if level == 0:
            image_show(gradient_scale_o, "gradient_norm")

        # local threshold.
        gradient_scale = gradient_scale_o > cv2.blur(gradient_scale_o, ksize=(block_size, block_size)) + thread_offset
        gradient_scale = gradient_scale.astype(np.uint8)

        morph_3x3 = cv2.filter2D(gradient_scale, cv2.CV_8U, morph_kernel_3x3)
        morph_5x5 = cv2.filter2D(gradient_scale, cv2.CV_8U, morph_kernel_5x5)
        isolation = np.zeros_like(gradient_scale_o)

        if dm_image.pyramid_factor[level] != 0:
            isolation[morph_3x3 < 9] = 1
            isolation[morph_5x5 <= 5] += 1
            isolation[gradient_scale == 1] += 1
            isolation[isolation != 3] = 0
            gradient_scale *= 255
            dilation_kernel = np.ones((2, 2))
            gradient_scale = cv2.dilate(gradient_scale, dilation_kernel, 1)
        else:
            gradient_scale *= 255

        rect_candidate = l_shape_finder(gradient_scale, rescale_factor=dm_image.pyramid_factor[level])
        for rect in rect_candidate:
            results.append(rect)

    results_after_nms = []
    box_iou_threshold = 0.95
    if len(results) > 0:
        results_after_nms.append(results[0])
        for i in range(1, len(results)):
            flag = True
            for res_rec in results_after_nms:
                if cal_iou(res_rec, results[i]) > box_iou_threshold:
                    flag = False
            if flag:
                results_after_nms.append(results[i])
    for rect in results_after_nms:
        cv2.drawContours(image_raw, [rect], 0, (255, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(dm_image.img_path) + 'edge.png'), image_raw)
    return results_after_nms


def l_shape_finder(src, rescale_factor=1, regression=False):
    h, w = src.shape[: 2]
    min_edge = min(h, w)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_candidate = []
    hull_candidate = []
    is_out = []

    for i in range(len(contours)):
        is_out.append(0)
        if hierarchy[0][i][3] == -1 or (hierarchy[0][i][3] != -1 and is_out[hierarchy[0][i][3]] == 0):
            is_out[i] = 1

    points = []
    for i in range(len(contours)):
        contour = contours[i]
        for point in contour:
            points.append(point)

    # coarse region finder by a convex hull based algorithm.
    for i in range(len(contours)):
        contour = contours[i]
        hie = hierarchy[0][i]
        # area_c = cv2.contourArea(contour)
        peri_c = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri_c, True)

        hull = cv2.convexHull(approx, clockwise=True)

        if (hie[2] == -1 or hierarchy[0][hie[2]][2] == -1) and is_out[i] == 1:
            hull_candidate.append(hull)
        area_h = cv2.contourArea(hull)
        peri_h = cv2.arcLength(hull, True)

        if peri_h < 0.3 * min_edge or is_out[i] == 0 or area_h == 0:
            continue

        # Assumption 1：All valid code area should be placed in the kind of 'center' of the picture.
        center_hull = get_center(hull)
        if min(min(center_hull[0], w - center_hull[0]), min(center_hull[1], h - center_hull[1])) < min_edge * 0.3:
            continue

        min_rect = cv2.minAreaRect(hull)
        min_rect = np.int0(cv2.boxPoints(min_rect))

        # for rect_point in min_rect:
        #     pass

        # Assumption 2：All valid area should have a proper aspect ratio.
        w_hull = calc_euclidean_distance(min_rect[0], min_rect[1])
        h_hull = calc_euclidean_distance(min_rect[1], min_rect[2])

        ratio = max(w_hull / h_hull, h_hull / w_hull)
        mask_hull = cv2.fillPoly(np.zeros_like(src), [hull], (255, 255, 255))
        area_c = np.sum(src * mask_hull)
        full_ratio = area_c / area_h
        if ratio > 4 or (full_ratio < 0.3):
            continue
        rect_candidate.append(min_rect)

    for i in range(len(rect_candidate)):
        rect_candidate[i] = rect_candidate[i] * rescale_factor
    for i in range(len(contours)):
        contours[i] = contours[i] * rescale_factor
    return rect_candidate

    # if regression:
    #     reg_candidate = []
    #     for rect in rect_candidate:
    #         summation = np.sum(rect, axis=1)
    #         max_idx = np.where(summation == max(summation))[0][0]
    #         rect_cp = rect.copy()
    #
    #         rect_cp[0] = rect[max_idx % 4]
    #         rect_cp[2] = rect[(max_idx - 2) % 4]
    #         if rect[(max_idx - 1) % 4][0] < rect[(max_idx - 3) % 4][0]:
    #             rect_cp[3] = rect[(max_idx - 1) % 4]
    #             rect_cp[1] = rect[(max_idx - 3) % 4]
    #         else:
    #             rect_cp[1] = rect[(max_idx - 1) % 4]
    #             rect_cp[3] = rect[(max_idx - 3) % 4]
    #         rect = rect_cp
    #
    #         for i in range(4):
    #             rect[i][0] = min(w - 1, max(0, rect[i][0]))
    #             rect[i][1] = min(h - 1, max(0, rect[i][1]))
    #
    #         # 2 - - 1
    #         # |     |
    #         # |     |
    #         # 3 - - 0
    #         for i in range(4):
    #             min_dist = 10000
    #             min_point = None
    #             for hull in hull_candidate:
    #                 for point in hull:
    #                     dist = calc_euclidean_distance(rect[i], point)
    #                     if min_dist > dist:
    #                         min_dist = dist
    #                         min_point = point
    #
    #             rect[i] = min_point
    #         reg_candidate.append(rect)
    #         rect_candidate = reg_candidate