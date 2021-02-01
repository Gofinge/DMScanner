import os
import numpy as np
import cv2
import math

from utils.vis import image_show
from utils.augmentation import normalize

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


def dm_entropy_detector(dm_image, output_dir, **kwargs):
    kernel_size = (3, 3)
    color_level = 5
    image = dm_image.gray.copy()
    image = cv2.pyrMeanShiftFiltering(src=dm_image.origin_img, sp=15, sr=40)
    # image = cv2.bilateralFilter(image, d=13, sigmaColor=30, sigmaSpace=8)
    image_show(image)
    # pixel_list = np.float32(image.reshape((-1)))
    # compactness, labels, centers = cv2.kmeans(pixel_list, color_level, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    # image = centers[labels.flatten()].reshape(dm_image.gray.shape)
    #
    # # image = cv2.bilateralFilter(image, d=13, sigmaColor=46, sigmaSpace=8)
    # image_show(image)
    #
    # h, w = image.shape[: 2]
    # entropy_map = np.zeros((h, w))
    #
    # half_kernel_size = [int(size / 2) for size in kernel_size]
    # for i in range(half_kernel_size[0], h - half_kernel_size[0]):
    #     for j in range(half_kernel_size[1], w - half_kernel_size[1]):
    #         entropy_map[i, j] = image_entropy(image[i-half_kernel_size[0]: i+half_kernel_size[0],
    #                                           j-half_kernel_size[1]: j+half_kernel_size[1]])
    # image_show(entropy_map)
    # entropy_list = np.float32(entropy_map.reshape((-1)))
    # compactness, labels, centers = cv2.kmeans(entropy_list, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8([0, 100, 200])
    # res = centers[labels.flatten()]
    # res2 = normalize(res.reshape(entropy_map.shape))
    # image_show(res2)
    # print("Pause")
    return

# def dm_entropy_detector(dm_image, output_dir, **kwargs):
#     kernel_size = (5, 5)
#     image = dm_image.gray
#     # image = cv2.bilateralFilter(image, d=13, sigmaColor=46, sigmaSpace=8)
#     image_show(image)
#     h, w = image.shape[: 2]
#     entropy_map = np.zeros((h, w))
#
#     half_kernel_size = [int(size / 2) for size in kernel_size]
#     for i in range(half_kernel_size[0], h - half_kernel_size[0]):
#         for j in range(half_kernel_size[1], w - half_kernel_size[1]):
#             entropy_map[i, j] = image_entropy(image[i-half_kernel_size[0]: i+half_kernel_size[0],
#                                               j-half_kernel_size[1]: j+half_kernel_size[1]])
#     image_show(entropy_map)
#     entropy_list = np.float32(entropy_map.reshape((-1)))
#     compactness, labels, centers = cv2.kmeans(entropy_list, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
#     centers = np.uint8([0, 100, 200])
#     res = centers[labels.flatten()]
#     res2 = normalize(res.reshape(entropy_map.shape))
#     image_show(res2)
#     print("Pause")
#     return


def image_entropy(src, num_level=256):
    h, w = src.shape[: 2]
    prob_list = [0 for i in range(num_level)]
    length = 256 / num_level
    for i in range(h):
        for j in range(w):
            val = src[i][j]
            level = int(val / length)
            prob_list[level] += 1
    prob_list = [float(number) / (h * w) for number in prob_list]
    entropy = 0
    for prob in prob_list:
        if prob != 0:
            entropy -= prob * math.log2(prob)
    return entropy


def image_entropy_cluster(src, num_cluster=2):
    prob_list = [0 for i in range(num_cluster)]
    src = np.float32(src.reshape((-1)))
    compactness, labels, centers = cv2.kmeans(src, num_cluster, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    for label in labels.flatten():
        prob_list[label] += 1
    prob_list = [float(number) / len(labels) for number in prob_list]
    entropy = 0
    for prob in prob_list:
        if prob != 0:
            entropy -= prob * math.log2(prob)
    return entropy
