"""
    本文件包含了图像增强预处理方法的类定义及相关参数设置
    Author:         Julian.Zhang
    Email:          julian.zhang@smartmore.com
    GrayScale:      GrayScale 类，用于包装灰度化方法类
    Binarizer:      Binarizer 类，用于包装二值化方法类
"""
import cv2
import numpy as np


class GrayScale:
    def __init__(self, method='general'):
        self.method = method
        self.local_window_size = 5

    def __call__(self, img, **kwargs):
        if self.method == 'human_modify':
            return manual_grayscale(img, **kwargs)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def manual_grayscale(input_img, **kwargs):
    weights = kwargs.get('weights')
    gray = np.zeros(input_img.shape[:2])
    for i in range(3):
        gray += input_img[:, :, i]*weights[i]
    return gray


class Binarizer:
    def __init__(self, method='general'):
        self.method = method
        self.local_window_size = 5

    def __call__(self, img):
        if self.method == 'local_bradley':
            return local_bradley(img, self.local_window_size)
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary


# bradley local threshold, also known as the zbar QR code thresholding method.
def local_bradley(input_img, window_size):
    h, w = input_img.shape

    s2 = window_size
    temperature = 15.0

    input_img = cv2.medianBlur(input_img, 3)

    # integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    # change: a dynamic programming method.
    int_img[0, 0] = input_img[0, 0]
    for row in range(h):
        for col in range(w):
            if col == 0:
                int_img[row, col] = input_img[row, col] + int_img[row-1, col]
            elif row == 0:
                int_img[row, col] = input_img[row, col] + int_img[row, col-1]
            else:
                int_img[row, col] = input_img[row, col] + int_img[row-1, col]\
                                    + int_img[row, col-1] - int_img[row-1, col-1]

    # output img
    out_img = np.zeros_like(input_img)
    for col in range(w):
        for row in range(h):
            # SxS region
            y0 = max(row - s2, 0)
            y1 = min(row + s2, h - 1)
            x0 = max(col - s2, 0)
            x1 = min(col + s2, w - 1)

            count = (y1-y0)*(x1-x0)

            sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]

            if input_img[row, col] * count < sum_ * (100. - temperature) / 100.:
                out_img[row, col] = 0
            else:
                out_img[row, col] = 255

    return out_img


def normalize(img):
    return cv2.normalize(np.absolute(img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
