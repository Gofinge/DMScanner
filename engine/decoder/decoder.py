from utils.registry import Registry
from data import DMImage

import numpy as np
import zxing
import cv2
import os

decoder = Registry()


class DMDecoder:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "warp"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "code"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "bor"), exist_ok=True)

    def decode(self, dm_image: DMImage):
        return decoder[self.cfg.METHOD](dm_image=dm_image, output_dir=self.output_dir, **self.cfg.ARG)


@decoder.register("line-cycle-zxing")
def dm_line_cycle_decoder(dm_image, output_dir, zxing_enable=True, **kwargs):
    img = dm_image.gray
    rets = dm_image.ret_candidate
    use_avaliable = np.zeros(len(rets))
    count = 0
    messages = []
    for ret in rets:
        # print(ret)
        show = False
        if use_avaliable[count] != 0:
            return messages
        use_avaliable[count] = 1
        pts1 = ret.astype(np.float32)
        width = int(np.sqrt((pts1[0][0] - pts1[1][0]) ** 2 + (pts1[0][1] - pts1[1][1]) ** 2))
        height = int(np.sqrt((pts1[2][0] - pts1[1][0]) ** 2 + (pts1[2][1] - pts1[1][1]) ** 2))
        aspect_ratio = float(width) / height
        if height > width:
            aspect_ratio = 1 / aspect_ratio
        pts2 = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
        # print(pts1, pts2)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (256, 256))

        search_width = 4
        search_range = 16
        internal_top, start_end_top, signal_top, num_top = get_signal(dst[:search_range, :], search_width)
        internal_btm, start_end_btm, signal_btm, num_btm = get_signal(dst[-search_range:, :], search_width)
        internal_lft, start_end_lft, signal_lft, num_lft = get_signal(np.transpose(dst[:, :search_range]),
                                                                      search_width)
        internal_rht, start_end_rht, signal_rht, num_rht = get_signal(np.transpose(dst[:, -search_range:]),
                                                                      search_width)

        # rectangle_case:
        # Lnon, Dmin, Dmax, Lmax/Lmin
        # Lnon, Lnon, Dmin, Dmax
        # Dmin, Dmax, Lmin, Lmax
        # Dmin, Lmin, Dmax, Lmax

        # Square conditions:
        # Lnon, Dash, Dash, Lmax
        # Lnon, Lnon, Dash, Dash
        # Dash, Dash, Lmax, Lmax
        # print(num_top, num_btm, num_lft, num_rht)
        four_egde = sorted([num_top, num_btm, num_lft, num_rht])
        sort_index = np.argsort(np.array([num_top, num_btm, num_lft, num_rht]))
        if four_egde[2] - four_egde[1] <= 1 and abs(four_egde[3] - 2 * four_egde[2]) <= 1:
            grid_num = round((four_egde[3] / 2 + four_egde[2] + four_egde[1]) / 3) * 2
            grid_num = [grid_num, grid_num]
        elif four_egde[3] - four_egde[2] <= 1 and four_egde[1] <= four_egde[2] // 4 and four_egde[2] >= 3:
            grid_num = round((four_egde[3] + four_egde[2]) / 2) * 2
            grid_num = [grid_num, grid_num]
        elif four_egde[1] - four_egde[0] <= 1 and abs(four_egde[2] - 2 * four_egde[1]) <= 1 and four_egde[0] >= 3:
            grid_num = round((four_egde[0] + four_egde[1]) / 2) * 2
            grid_num = [grid_num, grid_num]
        elif aspect_ratio > 1.6:
            if four_egde[1] > 3 and 4 >= four_egde[2] / four_egde[1] >= 2 and (
                    sort_index[2] + sort_index[1] != 5 or sort_index[2] + sort_index[1] != 1):
                # rectangle_case
                if sort_index[1] >= 2:
                    grid_num = [2 * int(four_egde[1] / 2), 2 * int(four_egde[2] / 2)]
                else:
                    grid_num = [2 * int(four_egde[2] / 2), 2 * int(four_egde[1] / 2)]
            elif four_egde[2] > 3 and 4 >= four_egde[3] / four_egde[2] >= 2 and (
                    sort_index[3] + sort_index[2] != 5 or sort_index[3] + sort_index[2] != 1):
                if sort_index[2] >= 2:
                    grid_num = [2 * int(four_egde[2] / 2), 2 * int(four_egde[3] / 2)]
                else:
                    grid_num = [2 * int(four_egde[3] / 2), 2 * int(four_egde[2] / 2)]
            elif four_egde[0] > 3 and 4 >= four_egde[1] / four_egde[0] >= 2 and (
                    sort_index[1] + sort_index[0] != 5 or sort_index[1] + sort_index[0] != 1):
                if sort_index[0] >= 2:
                    grid_num = [2 * int(four_egde[0] / 2), 2 * int(four_egde[1] / 2)]
                else:
                    grid_num = [2 * int(four_egde[1] / 2), 2 * int(four_egde[0] / 2)]
            else:
                # hardcode.
                if height > width:
                    grid_num = [8, 18]
                else:
                    grid_num = [18, 8]
        else:
            grid_num = [four_egde[3], four_egde[3]]

        inverse = False
        # draw a pseudo binary code.
        if min(grid_num) >= 4:
            grid_size_y = 256 / grid_num[0]
            grid_size_x = 256 / grid_num[1]

            res_img = np.zeros((8 * grid_num[0], 8 * grid_num[1]))
            mean_val = np.mean(dst)
            four_edge_gray_value = [np.mean(dst[3:int(grid_size_y - 3), :]),
                                    np.mean(dst[:, 3:int(grid_size_x - 3)]),
                                    np.mean(dst[int(grid_size_y * (grid_num[0] - 1) + 3):int(
                                        grid_size_y * grid_num[0] - 3), :]),
                                    np.mean(dst[:, int(grid_size_x * (grid_num[1] - 1) + 3):int(
                                        grid_size_x * grid_num[1] - 3)])]
            if abs(max(four_edge_gray_value) - mean_val) / ((abs(min(four_edge_gray_value) - mean_val)) + 0.1) > 3:
                inverse = True
            # print(inverse)
            for i in range(grid_num[0]):
                for j in range(grid_num[1]):
                    interest_area = dst[int(i * grid_size_y) + 3:int((i + 1) * grid_size_y) - 3,
                                    int(j * grid_size_x) + 3:int((j + 1) * grid_size_x) - 3]
                    if np.mean(interest_area) > mean_val:
                        if not inverse:
                            res_img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 255
                    elif inverse:
                        res_img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 255

            # refine finder boundary pattern.
            border_img = np.zeros_like(res_img)
            for i in range(1, grid_num[1], 2):
                border_img[-8:, i * 8:(i + 1) * 8] = 255
            for i in range(1, grid_num[0], 2):
                border_img[i * 8:(i + 1) * 8, -8:] = 255

            border_type = ['d', 'd', 'd', 'd']
            split_point = np.zeros(4)
            # top
            top_fake_border = res_img[0, :]
            split_point[0] = np.sum(np.abs(top_fake_border - np.concatenate((res_img[0, :-1], np.zeros(1)))) / 255)
            # bottom
            btm_fake_border = res_img[-1, :]
            split_point[1] = np.sum(np.abs(btm_fake_border - np.concatenate((res_img[-1, 1:], np.zeros(1)))) / 255)
            # left
            lft_fake_border = res_img[:, 0]
            split_point[2] = np.sum(np.abs(lft_fake_border - np.concatenate((res_img[:-1, 0], np.zeros(1)))) / 255)
            # right
            rht_fake_border = res_img[:, -1]
            split_point[3] = np.sum(np.abs(rht_fake_border - np.concatenate((res_img[1:, -1], np.zeros(1)))) / 255)
            index = np.argsort(split_point)
            # print(np.argsort(split_point), split_point)
            if index[-2] + index[-1] == 5 or index[-2] + index[-1] == 1:
                continue
            else:
                border_type[index[0]] = 'l'
                border_type[index[1]] = 'l'
                # print(border_type)
                if border_type[0] == border_type[3] == 'l':
                    border_img = cv2.flip(border_img, 1)
                elif border_type[1] == border_type[2] == 'l':
                    border_img = cv2.flip(border_img, 0)
                elif border_type[1] == border_type[3] == 'l':
                    border_img = cv2.transpose(border_img)

                res_img[-8:, :] = border_img[-8:, :]
                res_img[:8, :] = border_img[:8, :]
                res_img[:, -8:] = border_img[:, -8:]
                res_img[:, :8] = border_img[:, :8]
                res_img = cv2.copyMakeBorder(res_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
                # barcode = self.zxing_reader.decode()

        # draw some lines to justify the hypothesis.
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        # top
        if sorted([internal_top, internal_btm, internal_lft, internal_rht])[1] > 0:
            show = True
            # Draw grid.
            # if internal_top != 0:
            #     start_index = int(max(start_end_top[0] // internal_top, 0))
            #     for i in range(-start_index, 256):
            #         line = int(internal_top*i + start_end_top[0])
            #         if line > 256:
            #             break
            #         # dst = cv2.line(dst, (line, 0), (line, 256), (128, 255, 0), 2)
            # # bottom
            # if internal_btm != 0:
            #     start_index = int(max(start_end_top[0] // internal_btm, 0))
            #     for i in range(-start_index, 256):
            #         line = int(internal_btm*i + start_end_btm[0])
            #         if line > 256:
            #             break
            #         # dst = cv2.line(dst, (line, 0), (line, 256), (0, 128, 128), 2)
            # # left
            # if internal_lft != 0:
            #     start_index = int(max(start_end_top[0] // internal_lft, 0))
            #     for i in range(-start_index, 256):
            #         line = int(internal_lft*i + start_end_lft[0])
            #         if line > 256:
            #             break
            #         # dst = cv2.line(dst, (0, line), (256, line), (255, 0, 128), 2)
            # # right
            # if internal_rht != 0:
            #     start_index = int(max(start_end_top[0] // internal_rht, 0))
            #     for i in range(-start_index, 256):
            #         line = int(internal_rht*i + start_end_rht[0])
            #         if line > 256:
            #             break
            #         # dst = cv2.line(dst, (0, line), (256, line), (128, 40, 70), 2)
        #
        # print(img_data.img_path)
        if True:
            cv2.imwrite(output_dir + '/warp/{}_{}.png'.format(os.path.basename(dm_image.img_path), count), dst)
        if min(grid_num) >= 4:
            cv2.imwrite(output_dir + '/code/{}_{}.png'.format(os.path.basename(dm_image.img_path), count),
                        res_img)
            cv2.imwrite(output_dir + '/bor/{}_{}.png'.format(os.path.basename(dm_image.img_path), count),
                        border_img)
            if zxing_enable:
                message = zxing.BarCodeReader().decode(
                    output_dir + '/code/{}_{}.png'.format(os.path.basename(dm_image.img_path), count)
                )
                if message.raw != '':
                    messages.append(message.raw)
        count += 1
    return messages


def get_signal(img, search_width=4):
    res_interval = 0
    start_point = 0
    end_point = 256
    signals = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    show_flag = 0
    for i in range(img.shape[0] // search_width):
        signal = np.mean(img[i * search_width:(i + 1) * search_width], 0)
        signals.append(signal)
        max_amp = np.max(signal)
        min_amp = np.min(signal)
        avg_mean_amp = np.mean(signal)
        avg_mean_smooth = cv2.blur(signal, (1, 101))
        max_threshold = max_amp - 0.2 * (max_amp - min_amp) + avg_mean_smooth - avg_mean_amp
        min_threshold = min_amp + 0.2 * (max_amp - min_amp) + avg_mean_smooth - avg_mean_amp

        # store jump points.
        max_up_points = []
        min_up_points = []
        mean_up_points = []
        max_down_points = []
        min_down_points = []
        mean_down_points = []
        for i in range(len(signal) - 1):
            if signal[i] <= min_threshold[i] and signal[i + 1] > min_threshold[i]:
                min_up_points.append(i)
            elif signal[i] >= min_threshold[i] and signal[i + 1] < min_threshold[i]:
                min_down_points.append(i)
            elif signal[i] >= max_threshold[i] and signal[i + 1] < max_threshold[i]:
                max_down_points.append(i)
            elif signal[i] <= max_threshold[i] and signal[i + 1] > max_threshold[i]:
                max_up_points.append(i)
            elif signal[i] <= avg_mean_smooth[i] and signal[i + 1] > avg_mean_smooth[i]:
                mean_up_points.append(i)
            elif signal[i] >= avg_mean_smooth[i] and signal[i + 1] < avg_mean_smooth[i]:
                mean_down_points.append(i)

        final_intervals = []
        mean_1, res_1, point_1 = get_intervals(min_up_points)
        mean_2, res_2, point_2 = get_intervals(min_down_points)
        mean_3, res_3, point_3 = get_intervals(max_up_points)
        mean_4, res_4, point_4 = get_intervals(max_down_points)
        mean_5, res_5, point_5 = get_intervals(mean_up_points)
        mean_6, res_6, point_6 = get_intervals(mean_down_points)

        # assumption: the edge should have number of jump points,
        # and jump points should be organized.
        max_inter = max(mean_1, mean_2, mean_3, mean_4)  # , mean_5, mean_6)
        min_inter = min(mean_1, mean_2, mean_3, mean_4)  # , mean_5, mean_6)

        if min(len(res_1), len(res_2), len(res_3), len(res_4), len(res_5), len(res_6)) == 0 or \
                max_inter >= 64 or \
                max_inter - min_inter > min_inter / 3:
            continue

        max_diff = max(mean_1, mean_2, mean_3, mean_4) - min(mean_1, mean_2, mean_3, mean_4)
        interval_diff = max_inter - min_inter
        if max_diff < 2 and interval_diff <= 1:
            show_flag += 1
            res_interval = np.mean([mean_1, mean_2, mean_3, mean_4])
            num_interval = round(256 / res_interval)
            start_point = min(point_1[0], point_2[0], point_3[0], point_4[0], point_5[0], point_6[0])
            end_point = max(point_1[-1], point_2[-1], point_3[-1], point_4[-1], point_5[-1], point_6[-1])
            return res_interval, [start_point, end_point], signals, num_interval

    return res_interval, [start_point, end_point], signals, 0


def get_intervals(ls, tolerance=5):
    if len(ls) <= 1:
        return 0, [], []
    res = []
    point = []
    intervals = [ls[i + 1] - ls[i] for i in range(len(ls) - 1)]
    median = np.median(intervals)
    for i in range(len(intervals)):
        if abs(intervals[i] - median) <= tolerance:
            point.append(ls[i])
            res.append(intervals[i])

    if len(res) == 0:
        mean = 0
    else:
        mean = np.mean(res)
    return mean, res, point
