import cv2
import numpy as np

colors = np.array(
    [
        [43, 43, 200],
        [43, 106, 200],
        [43, 169, 200],
        [43, 200, 163],
        [43, 200, 101],
        [54, 200, 43],
        [116, 200, 43],
        [179, 200, 43],
        [200, 153, 43],
        [200, 90, 43],
        [200, 43, 64],
        [200, 43, 127],
        [200, 43, 190],
        [142, 43, 200],
        [80, 43, 200],
    ]
)


def image_show(src, name="image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
