import cv2


def image_show(src, name="image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()