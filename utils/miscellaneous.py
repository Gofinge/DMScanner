import cv2
import numpy as np
from tqdm import tqdm
import math
from scipy import spatial

# TODO add to config
BLUR_VALUE = 3
SQUARE_TOLERANCE = 0.15
AREA_TOLERANCE = 0.15
DISTANCE_TOLERANC = 0.15
WARP_DIM = 300
SMALL_DIM = 29


def count_children(hierarchy, parent, inner=False):
    if parent == -1:  # parent == -1的话就是没有父亲轮廓
        return 0
    elif not inner:
        return count_children(hierarchy, hierarchy[parent][2], True)
        # hierarchy[i][2]代表子轮廓
        # hierarchy[i][0]代表后一个轮廓
    return 1 + count_children(hierarchy, hierarchy[parent][0], True) + count_children(hierarchy, hierarchy[parent][2],
                                                                                      True)


def has_square_parent(hierarchy, squares, parent):
    if hierarchy[parent][3] == -1:  # 如果没有父轮廓，直接返回
        return False
    if hierarchy[parent][3] in squares:  # 如果父轮廓在squares中，说明有父轮廓
        return True
    return has_square_parent(hierarchy, squares, hierarchy[parent][3])  # 否则直接计算父轮廓


def get_center(c):
    m = cv2.moments(c)
    return [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])]


def get_angle(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return math.degrees(math.atan2(y_diff, x_diff))


def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def get_farthest_points(contour, center):  # 找到轮廓点的最远点的两个点
    distances = []
    distances_to_points = {}
    for point in contour:
        point = point[0]
        d = math.hypot(point[0] - center[0], point[1] - center[1])
        distances.append(d)
        distances_to_points[d] = point
    distances = sorted(distances)
    return [distances_to_points[distances[-1]], distances_to_points[distances[-2]]]


def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])  # 两个点的横坐标差异
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # 两个点的纵坐标差异

    def det(a, b):  # x_diff[0] * y_diff[1] - x_diff[1] * y_diff[0]
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return [-1, -1]

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return [int(x), int(y)]


def extend(a, b, length, int_represent=False):  #
    length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)  # 计算midpoint 和 center的距离
    if length_ab * length <= 0:
        return b
    result = [b[0] + (b[0] - a[0]) / length_ab * length, b[1] + (b[1] - a[1]) / length_ab * length]
    if int_represent:
        return [int(result[0]), int(result[1])]
    else:
        return result


def get_dist_P2L(PointP, Pointa, Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程
    A = 0
    B = 0
    C = 0
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]
    # 代入点到直线距离公式
    distance = 0
    distance = (A * PointP[0] + B * PointP[1] + C) / math.sqrt(A * A + B * B)

    return distance


def calc_euclidean_distance(point1, point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance


# 计算第四个点
def calc_fourth_point(point1, point2, point3):  # pint3为A点
    D = (point1[0] + point2[0] - point3[0], point1[1] + point2[1] - point3[1])
    return list(D)


# 三点构成一个三角形，利用两点之间的距离，判断邻边AB和AC,利用向量法以及平行四边形法则，可以求得第四个点D
def judge_beveling(point1, point2, point3):
    dist1 = calc_euclidean_distance(point1, point2)
    dist2 = calc_euclidean_distance(point1, point3)
    dist3 = calc_euclidean_distance(point2, point3)
    dist = [dist1, dist2, dist3]
    max_dist = dist.index(max(dist))
    if max_dist == 0:
        D = calc_fourth_point(point1, point2, point3)
    elif max_dist == 1:
        D = calc_fourth_point(point1, point3, point2)
    else:
        D = calc_fourth_point(point2, point3, point1)
    return D


def get_point(square, center1, center2):
    d_max = 0
    xi = 0
    yi = 0
    for i in range(4):
        # print("square[i][0]", square[i][0])
        d = np.abs(get_dist_P2L(square[i][0], center1, center2))
        # print("d", d)
        if d > d_max:
            d_max = d
            xi = square[i][0][0]
            yi = square[i][0][1]
    return [xi, yi]


def get_corner(main_square, south_square, east_square):
    main_center = get_center(main_square)
    # print("main_center:",main_center)
    south_center = get_center(south_square)
    # print("south_center:", south_center)
    east_center = get_center(east_square)
    # print("east_center:", east_center)
    sou_east_center = judge_beveling(main_center, south_center, east_center)
    # print("sou_east_center:", sou_east_center)
    main_corner = get_point(main_square, south_center, east_center)
    # print("main_corner:",main_corner)
    south_corner = get_point(south_square, main_center, sou_east_center)
    # print("south_corner:",south_corner)
    east_corner = get_point(east_square, main_center, sou_east_center)
    # print("east_corner:", east_corner)
    sou_east_corner = judge_beveling(main_corner, south_corner, east_corner)
    # print("sou_east_corner:", sou_east_corner)

    return [main_corner, south_corner, sou_east_corner, east_corner]


# 计算square内的黑白像素比
def calculate_proportion(binary, squares, i):
    background = np.zeros_like(binary)
    background = cv2.drawContours(background, squares, i, (255, 255, 255), 1)
    same_white = np.sum((background * binary > 0) * 1)
    area = cv2.contourArea(squares[i])
    return (area - same_white) / area


def trans_main(square, WARP_DIM):
    wrect = np.zeros((4, 2), dtype="float32")
    wrect[0] = square[0][0]
    wrect[1] = square[1][0]
    wrect[2] = square[2][0]
    wrect[3] = square[3][0]
    dst = np.array([
        [0, 0],
        [WARP_DIM - 1, 0],
        [WARP_DIM - 1, WARP_DIM - 1],
        [0, WARP_DIM - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(wrect, dst)
    return M, wrect


def trans_other(square, frame, wrect_center, WARP_DIM, M):
    wrect = np.zeros((4, 2), dtype="float32")
    wrect[0] = square[0][0]
    wrect[1] = square[1][0]
    wrect[2] = square[2][0]
    wrect[3] = square[3][0]
    Affine_img = cv2.warpAffine(frame, cv2.getAffineTransform(wrect[0:3, ::], wrect_center[0:3, ::]),
                                (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST)
    warp = cv2.warpPerspective(Affine_img, M, (WARP_DIM, WARP_DIM), flags=cv2.INTER_NEAREST)
    return warp


def get_cosine(vector_1, vector_2):
    return 1 - spatial.distance.cosine(vector_1, vector_2)


def connected_components_analysis(binary, area_tolerance):
    nums, area_map = cv2.connectedComponents(binary, 8)
    real_area = np.zeros_like(binary)
    for i in range(1, nums):
        if area_map[0, 0] == i or area_map[0, -1] == i or area_map[-1, 0] == i or area_map[-1, -1] == i:
            continue
        if np.sum(area_map == i) < area_tolerance:
            continue
        real_area = (area_map == i).astype(np.uint8) * 255 + real_area
    return real_area


def fill_blank(real_area):
    nums, area_map = cv2.connectedComponents(255 - real_area, 8)
    for i in range(1, nums):
        if np.sum(area_map == i) < 100:
            real_area[area_map == i] = 255
    return real_area


def norm_image(image):
    max_val = np.max(image)
    min_val = np.min(image)
    # print(max_val, min_val)
    return ((image - min_val)/(max_val - min_val)*255).astype(np.uint8)


def cal_iou(polygon_1, polygon_2):
    points = np.concatenate((polygon_1, polygon_2), axis=0)
    max_xy = np.max(points, axis=0) + 1
    min_xy = np.min(points, axis=0)
    blackboard_1 = np.ones((max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    blackboard_2 = np.ones((max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    poly_1 = [[[polygon_1[i][0] - min_xy[0], polygon_1[i][1] - min_xy[1]]] for i in range(len(polygon_1))]
    poly_2 = [[[polygon_2[i][0] - min_xy[0], polygon_2[i][1] - min_xy[1]]] for i in range(len(polygon_2))]
    poly_1 = np.array([list(poly_1)])
    poly_2 = np.array([list(poly_2)])
    cv2.fillPoly(blackboard_1, poly_1, color=(0, 0, 0))
    cv2.fillPoly(blackboard_2, poly_2, color=(0, 0, 0))
    summation = blackboard_1 + blackboard_2
    intersect = len(summation[summation == 0])
    union = np.sum(1 - blackboard_1 * blackboard_2)
    assert intersect / union <= 1.0
    return intersect / union


def generate_pyramid(src, num_pyramid_level):
    pyramid = []
    down = src.copy()
    for i in range(num_pyramid_level):
        if i == 0:
            pyramid.append(down)
        else:
            down = cv2.pyrDown(down)
            pyramid.append(down)
    return pyramid


def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c


def get_line_cross_point(line1, line2):
    # x1y1x2y2
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    # print(x, y)
    return [x, y]


def angle_points(point1, point2, point3, point4):
    vector_1 = np.array([point1[0][0] - point2[0][0],
                        point1[0][1] - point2[0][1]])
    vector_2 = np.array([point3[0][0] - point4[0][0],
                        point3[0][1] - point4[0][1]])
    cos = vector_1.dot(vector_2)/(np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    # print(point1, point2, point3, point4, cos)
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return np.arccos(cos)


def distance_points(point1, point2):
    return np.sqrt((point1[0][0] - point2[0][0])**2 +\
                   (point1[0][1] - point2[0][1])**2)


class TqdmBar(object):
    def __init__(self, data_loader,
                 total=0, description='', position=0, leave=False, use_bar=True):
        if use_bar:
            self.bar = tqdm(enumerate(data_loader), total=total, position=position, leave=leave)
            self.bar.set_description(description)
        else:
            self.bar = enumerate(data_loader)

    def set_postfix(self, info):
        if isinstance(self.bar, tqdm):
            self.bar.set_postfix(info)

    def close(self):
        if isinstance(self.bar, tqdm):
            self.bar.close()

    def clear(self, nolock=True):
        if isinstance(self.bar, tqdm):
            self.bar.clear(nolock=nolock)