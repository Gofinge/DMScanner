import os
import numpy as np
import cv2
import math
from utils.miscellaneous import get_center, calc_euclidean_distance, cal_iou, generate_pyramid, \
    get_line_cross_point, angle_points, distance_points, calc_fourth_point
from utils.augmentation import normalize
from utils.vis import image_show


def dm_gradient_edge_detector(dm_image,
                              output_dir,
                              num_pyramid_level=3,
                              block_size=150,
                              thread_offset=30,
                              nms_iou_threshold=0.95,
                              vis=False,
                              **kwargs):
    w_sobel_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    h_sobel_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    pyramid = generate_pyramid(dm_image.img, num_pyramid_level)

    results = []
    image_raw = dm_image.img

    for level in range(len(pyramid)):
        pyramid_factor = 2 ** level
        image = cv2.cvtColor(pyramid[level].copy(), cv2.COLOR_BGR2GRAY)
        image_smoothed = cv2.bilateralFilter(image, d=13, sigmaColor=46, sigmaSpace=8)
        # gradient
        gradient_w = cv2.filter2D(image_smoothed, cv2.CV_32F, w_sobel_kernel)
        gradient_h = cv2.filter2D(image_smoothed, cv2.CV_32F, h_sobel_kernel)
        gradient_scale_o = np.sqrt(gradient_w ** 2 + gradient_h ** 2)
        gradient_scale_o = normalize(gradient_scale_o)
        if vis:
            image_show(gradient_scale_o, "gradient_norm")

        # local threshold.
        gradient_scale = gradient_scale_o > cv2.blur(gradient_scale_o, ksize=(block_size, block_size)) + thread_offset
        gradient_scale = gradient_scale.astype(np.uint8)

        gradient_scale *= 255

        rect_candidate = l_shape_finder(gradient_scale, rescale_factor=pyramid_factor)
        for rect in rect_candidate:
            results.append(rect)

    results_after_nms = []
    if len(results) > 0:
        results_after_nms.append(results[0])
        for i in range(1, len(results)):
            flag = True
            for res_rec in results_after_nms:
                if cal_iou(res_rec, results[i]) > nms_iou_threshold:
                    flag = False
            if flag:
                results_after_nms.append(results[i])
    for rect in results_after_nms:
        cv2.drawContours(image_raw, [rect], 0, (255, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(dm_image.img_path) + 'edge.png'), image_raw)
    return results_after_nms


def l_shape_finder(src, rescale_factor=1):
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

    # coarse region finder by a convex hull based algorithm.
    for i in range(len(contours)):
        contour = contours[i]
        hie = hierarchy[0][i]
        approx = cv2.approxPolyDP(contour, (rescale_factor * min_edge) // 300, True)

        temp_contour = approx
        if len(temp_contour) > 4:
            max_length = 0
            second_length = 0
            point_max = 0
            point_second_max = 0
            for k in range(-1, len(temp_contour) - 1):
                length_pow = (temp_contour[k][0][0] - temp_contour[k + 1][0][0]) ** 2 + (
                            temp_contour[k][0][1] - temp_contour[k + 1][0][1]) ** 2
                if max_length < length_pow:
                    second_length = max_length
                    max_length = length_pow
                    point_second_max = point_max
                    point_max = k
                elif second_length < length_pow:
                    second_length = length_pow
                    point_second_max = k

            line_1 = [temp_contour[point_max][0][0], temp_contour[point_max][0][1],
                      temp_contour[point_max + 1][0][0], temp_contour[point_max + 1][0][1]]
            line_2 = [temp_contour[point_second_max][0][0], temp_contour[point_second_max][0][1],
                      temp_contour[point_second_max + 1][0][0], temp_contour[point_second_max + 1][0][1]]

            angle = angle_points(temp_contour[point_max], temp_contour[point_max + 1],
                                 temp_contour[point_second_max], temp_contour[point_second_max + 1])

            if np.sqrt(second_length) * rescale_factor > 50 and 1.4 < angle < 1.8:

                cross = [get_line_cross_point(line_1, line_2)]
                if distance_points(temp_contour[point_max], cross) > distance_points(temp_contour[point_max + 1],
                                                                                     cross):
                    far_point_one = temp_contour[point_max]
                else:
                    far_point_one = temp_contour[point_max + 1]
                if distance_points(temp_contour[point_second_max], cross) > distance_points(
                        temp_contour[point_second_max + 1], cross):
                    far_point_two = temp_contour[point_second_max]
                else:
                    far_point_two = temp_contour[point_second_max + 1]
                rect = np.array([far_point_one[0],
                                 cross[0],
                                 far_point_two[0],
                                 calc_fourth_point(far_point_one[0], far_point_two[0], cross[0])
                                 ]).astype(np.int32)
                rect_candidate.append(rect)

        hull = cv2.convexHull(approx, clockwise=True)

        if (hie[2] == -1 or hierarchy[0][hie[2]][2] == -1) and is_out[i] == 1:
            hull_candidate.append(hull)
        area_h = cv2.contourArea(hull)
        peri_h = cv2.arcLength(hull, True)

        if peri_h < 0.5 * min_edge or is_out[i] == 0 or area_h == 0:
            continue

        # Assumption 1：All valid code area should be placed in the kind of 'center' of the picture.
        center_hull = get_center(hull)
        if min(min(center_hull[0], w - center_hull[0]), min(center_hull[1], h - center_hull[1])) < min_edge * 0.3:
            continue

        min_rect = cv2.minAreaRect(hull)
        min_rect = np.int0(cv2.boxPoints(min_rect))

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
