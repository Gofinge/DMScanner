import os
import cv2
import numpy as np

from data import DMImage, DMatrix
from utils.registry import Registry
from utils.miscellaneous import get_center, calc_euclidean_distance, cal_iou
from utils.augmentation import normalize

detector = Registry()


class DMDetector:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir

    def detect(self, dm_image: DMImage):
        return detector[self.cfg.METHOD](dm_image=dm_image, output_dir=self.output_dir, **self.cfg.ARG)


# TODO
@detector.register("general")
def dm_general_detector(dm_image, output_dir, **kwargs):
    # A computation friendly way.
    gray_pyramid = dm_image.gray_pyramid
    augmented_img = None
    augmented_inverse_img = None
    for level in range(len(gray_pyramid) - 1, -1, -1):
        search_img = gray_pyramid[level]
        search_img = cv2.bilateralFilter(search_img, 13, 46, 8)
        inverse_img = 255 - search_img
        h, w = search_img.shape[:2]
        log_img = np.log(search_img.astype(np.float32) + 1)
        low_pass = cv2.blur(log_img, ksize=(41, 41))
        brightness = (8.0 - low_pass)
        augmented_img = (np.exp(log_img * 0.4 + brightness * 0.6) - 1).astype(np.uint8)
        log_inverse_img = np.log(inverse_img.astype(np.float32) + 1)
        low_inverse_pass = cv2.blur(log_inverse_img, ksize=(41, 41))
        inverse_brightness = (8.0 - low_inverse_pass)
        augmented_inverse_img = (np.exp(log_inverse_img * 0.4 + inverse_brightness * 0.6) - 1).astype(np.uint8)
        break
    augmented_img = cv2.Canny(augmented_img, 5, 20)
    augmented_inverse_img = cv2.Canny(augmented_inverse_img, 5, 20)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(dm_image.img_path) + 'edge.png'), augmented_inverse_img)
    return DMatrix()


@detector.register("gradient-edge")
def dm_gradient_edge_detector(dm_image, **kwargs):
    # gradient + edge detect
    x_sobel_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    y_sobel_kernel = np.array([[-1, -2, -1],
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
    filterSize = (25, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)
    # x_isolation_kernel = np.array([[1,1,1,0,0]])
    # y_isolation_kernel = np.array([[1],[1],[1],[0],[0]])
    block_size = 35
    # print(block_size)
    thre_offset = 15

    gray_pyramid = dm_image.gray_pyramid
    result = []
    result_img = cv2.cvtColor(dm_image.gray, cv2.COLOR_GRAY2BGR)
    result_comp_img = result_img.copy()
    res_line = np.zeros((0, 1, 4))
    res_simplifyed = np.zeros((0, 1, 4))
    for level in range(len(gray_pyramid)):
        search_img = gray_pyramid[level]
        smooth = cv2.bilateralFilter(search_img, 13, 46, 8)
        # gradient amptitude.
        gradient_x = cv2.filter2D(smooth, cv2.CV_32F, x_sobel_kernel)
        gradient_y = cv2.filter2D(smooth, cv2.CV_32F, y_sobel_kernel)
        gradient_scale_o = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_scale_o[gradient_scale_o > 255] = 255
        gradient_scale_o = normalize(gradient_scale_o)
        # remove isolation points in a small neighborhood. --stage1
        # gradient_scale_blur = cv2.GaussianBlur(gradient_scale_o, (7, 7), 0)
        # gradient_scale_o[gradient_scale_o > gradient_scale_blur] = gradient_scale_blur[gradient_scale_o > gradient_scale_blur]

        # local threshold.
        gradient_scale = gradient_scale_o > cv2.blur(gradient_scale_o, ksize=(block_size, block_size)) + thre_offset
        gradient_scale = (gradient_scale).astype(np.uint8)

        # remove isolation points in a small neighborhood. --stage2
        morph_3x3 = cv2.filter2D(gradient_scale, cv2.CV_8U, morph_kernel_3x3)
        morph_5x5 = cv2.filter2D(gradient_scale, cv2.CV_8U, morph_kernel_5x5)
        isolation = np.zeros_like(gradient_scale_o)

        if dm_image.pyramid_factor[level] != 0:
            isolation[morph_3x3 < 9] = 1
            isolation[morph_5x5 <= 5] += 1
            isolation[gradient_scale == 1] += 1
            # print(np.unique(morph_3x3), np.unique(isolation))
            isolation[isolation != 3] = 0
            # isolation /= 3
            isolation = cv2.dilate(isolation, np.ones((2, 2)), 1)
            # gradient_scale[isolation > 0] = 0
            gradient_scale *= 255
            dilation_kernel = np.ones((2, 2))
            gradient_scale = cv2.dilate(gradient_scale, dilation_kernel, 1)
            # gradient_scale[isolation > 0] = 0
        else:
            gradient_scale *= 255

        if False:
            line = detector.detect(gradient_scale_o)
            if line is not None:
                simplifyed_line = line_combine(line, smooth)
                line = line * dm_image.pyramid_factor[level]
                simplifyed_line = simplifyed_line * dm_image.pyramid_factor[level]

                res_line = np.concatenate((res_line, line))
                res_simplifyed = np.concatenate((res_simplifyed, simplifyed_line))
                # print(res_simplifyed.shape, line.shape)
            result_img = detector.drawSegments(result_img, res_simplifyed)
            result_comp_img = detector.drawSegments(result_comp_img, res_line)
        tophat_img = cv2.morphologyEx(gradient_scale,
                                      cv2.MORPH_TOPHAT,
                                      kernel)
        rect_candidate = l_shape_finder(gradient_scale, rescale_factor=dm_image.pyramid_factor[level])

        # print(rect_candidate)
        for rect in rect_candidate:
            result.append(rect)
    # simple filtering for overlap.

    result_after_nms = []
    box_iou_threshold = 0.95
    if len(result) > 0:
        result_after_nms.append(result[0])
        for i in range(1, len(result)):
            flag = True
            for res_rec in result_after_nms:
                if cal_iou(res_rec, result[i]) > box_iou_threshold:
                    flag = False
            if flag:
                result_after_nms.append(result[i])

    gradient_scale = cv2.cvtColor(gradient_scale, cv2.COLOR_GRAY2BGR)
    for rect in result_after_nms:
        cv2.drawContours(gradient_scale, [rect], 0, (255, 255, 0), 1)
    # cv2.imwrite(os.path.join(out_dir, os.path.basename(dm_image.img_path) + 'segment_merge.png'), result_img)
    # cv2.imwrite(os.path.join(out_dir, os.path.basename(dm_image.img_path) + 'segment_origin.png'), result_comp_img)
    # cv2.imwrite(os.path.join(out_dir, os.path.basename(dm_image.img_path) + 'edge.png'), gradient_scale)
    # print(result_after_nms)
    return result_after_nms


@detector.register("max-min")
def dm_max_min_detector(img_data, **kwargs):
    # Max-min Canny process.
    win_s = 3
    data = img_data.gray
    data = cv2.medianBlur(data, 5)
    h, w = data.shape[:2]
    n_h = int(h / win_s)
    n_w = int(w / win_s)
    inter_max = np.zeros_like(data)
    inter_min = np.zeros_like(data)
    inter_col_max = np.zeros_like(data)
    inter_col_min = np.zeros_like(data)

    for i in range(h):
        for j in range(w):
            interest_seg = data[max(i - win_s, 0):min(i + win_s, h), j]
            inter_col_max[i, j] = np.max(interest_seg)
            inter_col_min[i, j] = np.min(interest_seg)
    for i in range(h):
        for j in range(w):
            interest_seg_max = inter_col_max[i, max(j - win_s, 0):min(j + win_s, w)]
            interest_seg_min = inter_col_min[i, max(j - win_s, 0):min(j + win_s, w)]
            inter_max[i, j] = np.max(interest_seg_max)
            inter_min[i, j] = np.min(interest_seg_min)

    max_min = inter_max - inter_min
    dense = cv2.GaussianBlur(max_min, (3, 3), 0)
    # cv2.imwrite(os.path.join(out_dir, os.path.basename(img_data.img_path) + 'max_min.png'), max_min)
    _, max_min = cv2.threshold(dense, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # canny, rect_candidate = recur_rect_finder(max_min)
    canny, rect_candidate = l_shape_finder(max_min)
    dense = cv2.cvtColor(dense, cv2.COLOR_GRAY2BGR)
    for rect in rect_candidate:
        cv2.drawContours(dense, [rect], 0, (0, 255, 0), 2)
    # cv2.imwrite(os.path.join(out_dir, os.path.basename(img_data.img_path) + 'canny.png'), canny)
    return rect_candidate


def l_shape_finder(data, rescale_factor=1, regression=False, **kwargs):
    h, w = data.shape[:2]
    min_edge = min(h, w)
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_candidate = []
    hull_candidate = []
    is_out = []

    for i in range(len(contours)):
        is_out.append(0)
        if hierarchy[0][i][3] == -1 or (hierarchy[0][i][3] != -1 and is_out[hierarchy[0][i][3]] == 0):
            is_out[i] = 1

    # traverse point.
    # 20210104 TODO.
    points = []
    for i in range(len(contours)):
        contour = contours[i]
        for point in contour:
            points.append(point)

    # coarse region finder by a convex hull based algorithm.
    for i in range(len(contours)):
        contour = contours[i]
        hie = hierarchy[0][i]
        area_c = cv2.contourArea(contour)
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
        mask_hull = cv2.fillPoly(np.zeros_like(data), [hull], (255, 255, 255))
        area_c = np.sum(data * mask_hull)
        full_ratio = area_c / area_h
        if ratio > 4 or (full_ratio < 0.3):
            continue
        rect_candidate.append(min_rect)
    # print(rect_candidate)

    # simple regression: find a nearest point in hulls.
    if regression:
        reg_candidate = []
        for rect in rect_candidate:
            summation = np.sum(rect, axis=1)
            max_idx = np.where(summation == max(summation))[0][0]
            rect_cp = rect.copy()

            rect_cp[0] = rect[max_idx % 4]
            rect_cp[2] = rect[(max_idx - 2) % 4]
            if rect[(max_idx - 1) % 4][0] < rect[(max_idx - 3) % 4][0]:
                rect_cp[3] = rect[(max_idx - 1) % 4]
                rect_cp[1] = rect[(max_idx - 3) % 4]
            else:
                rect_cp[1] = rect[(max_idx - 1) % 4]
                rect_cp[3] = rect[(max_idx - 3) % 4]
            rect = rect_cp

            for i in range(4):
                rect[i][0] = min(w - 1, max(0, rect[i][0]))
                rect[i][1] = min(h - 1, max(0, rect[i][1]))

            # 2 - - 1
            # |     |
            # |     |
            # 3 - - 0
            for i in range(4):
                min_dist = 10000
                min_point = None
                for hull in hull_candidate:
                    for point in hull:
                        dist = calc_euclidean_distance(rect[i], point)
                        if min_dist > dist:
                            min_dist = dist
                            min_point = point

                rect[i] = min_point
            reg_candidate.append(rect)
            rect_candidate = reg_candidate
    for i in range(len(rect_candidate)):
        rect_candidate[i] = rect_candidate[i] * rescale_factor
    for i in range(len(contours)):
        contours[i] = contours[i] * rescale_factor

    return rect_candidate
