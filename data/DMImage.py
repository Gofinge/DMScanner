import cv2
import numpy as np
from utils.augmentation import Binarizer, GrayScale


class DMImage:
    """
        Class to define the test image.
    """

    def __init__(self, img_path, cfg=None):
        self.cfg = cfg
        self.img_path = img_path
        self.img = cv2.imread(self.img_path, 1)
        self.img_size = self.img.shape[:2]
        if self.img is None:
            return
        # record origin image
        self.origin_img = self.img.copy()
        self.origin_size = self.origin_img.shape[:2]

        # parameters
        # TODO: check usage
        self.is_DPM = False
        self.illuminance_reverse = False
        self.ret_candidate = []

        # process image
        self.restrict_resize()
        self.binarizer = Binarizer()
        self.binarizer.local_window_size = self.cfg.BINARIZER.LOCAL_WINDOWS_SIZE
        self.binarizer.method = self.cfg.BINARIZER.METHOD
        self.grayscale = GrayScale()
        self.grayscale.method = self.cfg.GRAYSCALE.METHOD
        self.gray = self.grayscale(self.img)
        self.binary = self.binarizer(self.gray)
        self.pyramid_factor = None
        self.gray_pyramid, self.binary_pyramid = self.update_pyramid(self.cfg.PYRAMID.FACTOR_LIST)

    def restrict_resize(self):
        # resize image to fit size limitation, priority maximum
        img_size = np.array(self.img.shape[:2])
        idx_order = img_size.argsort()
        if img_size[idx_order[0]] < self.cfg.MIN_SIZE:
            resize_rate = self.cfg.MIN_SIZE / img_size[idx_order[0]]
            img_size = (img_size * resize_rate).astype(int)
            self.img = cv2.resize(self.img, tuple(img_size),
                                  interpolation=cv2.INTER_LINEAR)

        if img_size[1] > self.cfg.MAX_SIZE:
            resize_rate = self.cfg.MAX_SIZE / img_size[idx_order[1]]
            img_size = (img_size * resize_rate).astype(int)
            self.img = cv2.resize(self.img, tuple(img_size),
                                  interpolation=cv2.INTER_LINEAR)
        self.img_size = self.img.shape[:2]
        return

    def update_pyramid(self, factor_list):
        assert min(self.img_size) > max(factor_list)
        gray_pyramid = []
        binary_pyramid = []
        self.pyramid_factor = factor_list
        for i in self.pyramid_factor:
            gray_pyramid.append(cv2.resize(self.gray, (0, 0), fx=1 / float(i), fy=1 / float(i),
                                           interpolation=cv2.INTER_LINEAR))
            binary_pyramid.append(
                cv2.resize(self.binary, (0, 0), fx=1 / float(i), fy=1 / float(i), interpolation=cv2.INTER_NEAREST))
        return gray_pyramid, binary_pyramid




