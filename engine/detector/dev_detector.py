import os
import numpy as np
import cv2

from utils.vis import image_show


def dm_dev_detector(dm_image, output_dir, **kwargs):
    image = inverse_color(dm_image.gray.copy())
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    points = detector.detect(image)
    imag = cv2.drawKeypoints(dm_image.img.copy(), points, dm_image.img.copy(), color=(255, 0, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    image_show(imag)
    print("Pause")
    return


def inverse_color(src):
    height, width = src.shape
    for row in range(height):
        for col in range(width):
            src[row, col] = 255 - src[row, col]
    return src

