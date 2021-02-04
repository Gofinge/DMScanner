import cv2
import numpy as np
from utils.augmentation import Binarizer, GrayScale


class DMImage:
    """
        Class to define the test image.
    """

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path, 1)

        if self.img is None:
            return

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_size = self.img.shape[:2]

        self.ret_candidate = []





