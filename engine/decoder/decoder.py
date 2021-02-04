from utils.registry import Registry
from data import DMImage

from scipy.fftpack import fft
import numpy as np
import zxing
import cv2
import os

decoder = Registry()

POSSIBLE_RECT_SIZE = np.array([[8, 18], [12, 26], [16, 36]])

POSSIBLE_SQAR_SIZE = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 32,
                               36, 40, 44, 48, 52, 64, 72, 80, 88, 96,
                               104, 120, 132, 144])

class DMDecoder:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "warp"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "recon"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "mirror"), exist_ok=True)

    def decode(self, dm_image: DMImage):
        return decoder[self.cfg.METHOD](dm_image=dm_image, output_dir=self.output_dir, **self.cfg.ARG)


@decoder.register("line-cycle-zxing")
def dm_line_cycle_decoder( dm_image, output_dir, zxing_enable=True, **kwargs):
    img = cv2.GaussianBlur(dm_image.gray, (3, 3), 0)
    rets = dm_image.ret_candidate
    use_avaliable = np.zeros(len(rets))
    count = 0
    messages = []

    for ret in rets:
        show = False
        if use_avaliable[count] != 0:
            return
        use_avaliable[count] = 1
        pts1 = ret.astype(np.float32)
        width = int(np.sqrt((pts1[0][0] - pts1[1][0]) ** 2 + (pts1[0][1] - pts1[1][1]) ** 2))
        height = int(np.sqrt((pts1[2][0] - pts1[1][0]) ** 2 + (pts1[2][1] - pts1[1][1]) ** 2))
        aspect_ratio = float(width) / height
        if aspect_ratio < 1:
            width = min(256, max(128, width))
            height = int(width / aspect_ratio)
        else:
            height = min(256, max(128, height))
            width = int(height * aspect_ratio)
        # aspect_ratio = float(width)/height
        if height > width:
            aspect_ratio = 1 / aspect_ratio
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (width, height))
        search_range_w = width // 10
        search_width_w = max(4, search_range_w // 4)
        search_range_h = height // 10
        search_width_h = max(4, search_range_h // 4)

        internal_top, num_top = get_signal(dst[:search_range_w, :], width, search_width_w)
        internal_btm, num_btm = get_signal(dst[-search_range_w:, :], width, search_width_w)
        internal_lft, num_lft = get_signal(np.transpose(dst[:, :search_range_h]), height, search_width_h)
        internal_rht, num_rht = get_signal(np.transpose(dst[:, -search_range_h:]), height, search_width_h)

        four_egde = sorted([num_top, num_btm, num_lft, num_rht])
        sort_index = np.argsort(np.array([num_top, num_btm, num_lft, num_rht]))

        if aspect_ratio <= 1.6:
            # 方形重写。
            large_ud = max(num_top, num_btm)
            small_ud = min(num_top, num_btm)
            large_lr = max(num_lft, num_rht)
            small_lr = min(num_lft, num_rht)

            if (abs(large_ud - small_ud * 2) == 0 or abs(
                    large_ud - small_lr * 2) == 0) and large_ud >= 10:  # 上下最多跳跃点为正解时
                num = POSSIBLE_SQAR_SIZE[np.argmin(abs(POSSIBLE_SQAR_SIZE - large_ud))]
            elif (abs(large_lr - small_lr * 2) == 0 or abs(
                    large_lr - small_ud * 2) == 0) and large_lr >= 10:  # 左右最多跳跃点为正解时
                num = POSSIBLE_SQAR_SIZE[np.argmin(abs(POSSIBLE_SQAR_SIZE - large_lr))]
            elif (abs(large_ud - large_lr * 2) == 0 or abs(large_lr - large_ud * 2) == 0) and max(large_lr,
                                                                                                  large_ud) >= 10:  # 存在双倍关系时
                num = POSSIBLE_SQAR_SIZE[np.argmin(abs(POSSIBLE_SQAR_SIZE - max(large_ud, large_lr)))]
            else:
                if large_ud > large_lr:
                    index = np.argmin(
                        abs(np.concatenate((POSSIBLE_SQAR_SIZE - large_ud * 2, POSSIBLE_SQAR_SIZE - large_ud)))) % \
                            POSSIBLE_SQAR_SIZE.shape[0]
                    num = POSSIBLE_SQAR_SIZE[index]
                else:
                    index = np.argmin(
                        abs(np.concatenate((POSSIBLE_SQAR_SIZE - large_lr * 2, POSSIBLE_SQAR_SIZE - large_lr)))) % \
                            POSSIBLE_SQAR_SIZE.shape[0]
                    num = POSSIBLE_SQAR_SIZE[index]
            grid_num = [num, num]

        elif aspect_ratio > 1.6:
            # 矩形重写。
            # ud: up down
            # lr: left right
            large_ud = max(num_top, num_btm)
            small_ud = min(num_top, num_btm)
            large_lr = max(num_lft, num_rht)
            small_lr = min(num_lft, num_rht)
            if width > height:
                # #-------------#
                # |             |
                # #-------------#
                # 长边的点数量筛选。

                if abs(large_ud - small_ud * 2) == 0:
                    grid_num = POSSIBLE_RECT_SIZE[np.argmin(abs(POSSIBLE_RECT_SIZE[:, 1] - large_ud))]

                elif abs(large_lr - small_lr * 2) == 0:
                    grid_num = POSSIBLE_RECT_SIZE[np.argmin(abs(POSSIBLE_RECT_SIZE[:, 0] - large_lr))]

                else:  # 最差的结果:直接取长边。
                    index = np.argmin(abs(np.concatenate(
                        (POSSIBLE_RECT_SIZE[:, 1] - large_ud * 2, POSSIBLE_RECT_SIZE[:, 1] - large_ud)))) % \
                            POSSIBLE_RECT_SIZE.shape[0]
                    grid_num = POSSIBLE_RECT_SIZE[index]
            else:
                # #----#
                # |    |
                # |    |
                # |    |
                # |    |
                # #----#
                if abs(large_lr - small_lr * 2) == 0:
                    grid_num = POSSIBLE_RECT_SIZE[np.argmin(abs(POSSIBLE_RECT_SIZE[:, 1] - large_lr))]

                elif abs(large_ud - small_ud * 2) == 0:
                    grid_num = POSSIBLE_RECT_SIZE[np.argmin(abs(POSSIBLE_RECT_SIZE[:, 0] - large_ud))]

                else:  # 最差的结果:直接取长边。
                    index = np.argmin(abs(np.concatenate(
                        (POSSIBLE_RECT_SIZE[:, 1] - large_lr * 2, POSSIBLE_RECT_SIZE[:, 1] - large_lr)))) % \
                            POSSIBLE_RECT_SIZE.shape[0]
                    grid_num = POSSIBLE_RECT_SIZE[index]

                grid_num = [grid_num[1], grid_num[0]]

        inverse = False
        # draw a pseudo binary code.
        if min(grid_num) >= 4:
            grid_size_y = height / grid_num[0]
            grid_size_x = width / grid_num[1]
            y_offset = int(grid_size_y // 5)
            x_offset = int(grid_size_x // 5)
            res_img = np.zeros((8 * grid_num[0], 8 * grid_num[1]))
            sample_val = np.zeros((grid_num[0], grid_num[1]))

            # 取样开始。
            for i in range(grid_num[0]):
                for j in range(grid_num[1]):
                    sample_val[i][j] = np.mean(
                        dst[int(i * grid_size_y) + y_offset:int((i + 1) * grid_size_y) - y_offset,
                        int(j * grid_size_x) + x_offset:int((j + 1) * grid_size_x) - x_offset])

            mean_val = np.mean(sample_val)
            mean_border = (np.mean(sample_val[0, :]) + np.mean(sample_val[:, 0]) +
                           np.mean(sample_val[-1, :]) + np.mean(sample_val[:, -1])) / 4
            # DOTO: 是否是dash需要判断。
            num_white_points = len(sample_val[0, :][sample_val[0, :] > mean_border]) + \
                               len(sample_val[:, 0][sample_val[:, 0] > mean_border]) + \
                               len(sample_val[-1, :][sample_val[-1, :] > mean_border]) + \
                               len(sample_val[:, -1][sample_val[:, -1] > mean_border])
            if num_white_points > grid_num[0] + grid_num[1]:
                inverse = True

            for i in range(grid_num[0]):
                for j in range(grid_num[1]):
                    interest_area = sample_val[i][j]
                    if np.mean(interest_area) > mean_val:
                        if not inverse:
                            res_img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 255
                    elif inverse:
                        res_img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 255
            # cv2.imwrite(out_dir + '/warp/init_res{}_{}.png'.format(os.path.basename(img_data.img_path), count), res_img)
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
            split_point[0] = np.sum(np.abs(top_fake_border - np.concatenate((res_img[0, 1:], np.zeros(1)))) / 255)
            # bottom
            btm_fake_border = res_img[-1, :]
            split_point[1] = np.sum(np.abs(btm_fake_border - np.concatenate((res_img[-1, 1:], np.zeros(1)))) / 255)
            # left
            lft_fake_border = res_img[:, 0]
            split_point[2] = np.sum(np.abs(lft_fake_border - np.concatenate((res_img[1:, 0], np.zeros(1)))) / 255)
            # right
            rht_fake_border = res_img[:, -1]
            split_point[3] = np.sum(np.abs(rht_fake_border - np.concatenate((res_img[1:, -1], np.zeros(1)))) / 255)
            index = np.argsort(split_point)
            # print(np.argsort(split_point), split_point)

            border_type[index[0]] = 'l'
            border_type[index[1]] = 'l'
            if border_type[0] == border_type[3] == 'l':
                border_img = cv2.flip(border_img, 1)
            elif border_type[1] == border_type[2] == 'l':
                border_img = cv2.flip(border_img, 0)
            elif border_type[1] == border_type[3] == 'l':
                # print('trans')
                border_img = cv2.flip(border_img, 1)
                border_img = cv2.flip(border_img, 0)

            res_img[-8:, :] = border_img[-8:, :]
            res_img[:8, :] = border_img[:8, :]
            res_img[:, -8:] = border_img[:, -8:]
            res_img[:, :8] = border_img[:, :8]
            res_img = cv2.copyMakeBorder(res_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
            # barcode = self.zxing_reader.decode()
        # draw some lines to justify the hypothesis.
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        # print(output_dir + '/warp/{}_{}.png'.format(os.path.basename(dm_image.img_path), count))
        cv2.imwrite(output_dir + '/warp/{}_{}.png'.format(os.path.basename(dm_image.img_path), count), dst)
        if min(grid_num) >= 4:
            cv2.imwrite(output_dir + '/recon/{}_{}_code.png'.format(os.path.basename(dm_image.img_path), count),
                        res_img)
            cv2.imwrite(output_dir + '/mirror/{}_{}_code.png'.format(os.path.basename(dm_image.img_path), count),
                        cv2.flip(res_img, 0))
            if zxing_enable:
                message = zxing.BarCodeReader().decode(
                    output_dir + '/recon/{}_{}_code.png'.format(os.path.basename(dm_image.img_path), count)
                )
                if message.raw != '':
                    return message.raw
        count += 1
    return None


def get_signal(img, length, search_width=4):
    fft_vote = np.zeros(length)
    signal = []
    for i in range(img.shape[0] // search_width):
        signal = np.mean(img[i * search_width:(i + 1) * search_width], 0)
        sig_fft_y = fft(signal)
        fft_vote += np.abs(sig_fft_y)

    candidates = cv2.flip(np.argsort(fft_vote[4:length // 2]), 0)[0:5] + 4
    return length / candidates[0][0], candidates[0][0]


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
