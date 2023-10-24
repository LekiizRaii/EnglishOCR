import matplotlib.pyplot as plt
from PIL import Image
from utils import *


class CharacterSegmentation:
    def __init__(self, img_dir, seperation_threshold=15, round_threshold=50, scan_width=3):
        """
        Parameters:
        - img_dir (string): Directory to the image
        - stroke_threshold (int): Threshold to be a stroke
        - seperation_threshold (int): Threshold to be a valid seperation
        - round_threshold (int): Threshold to be a round region

        Returns:
        """

        # Open image
        self.path = img_dir
        self.img = Image.open(self.path).convert('L')
        self.img = np.asarray(self.img, dtype='float64')

        # Image height and width
        self.h, self.w = self.img.shape

        # Binarize the image
        mask = (self.img / 255) > 0.8
        self.img[mask] = 0.0
        self.img[~mask] = 1.0

        # Set other parameters
        self.scan_width = scan_width
        self.stroke_width = self.cal_stroke_width()
        self. seperation_threshold = seperation_threshold
        self.round_threshold = round_threshold

    def cal_stroke_width(self):
        stroke_list = list()
        width = 0
        for i in range(self.h):
            for j in range(self.w):
                if self.img[i, j] == 1.0:
                    width += 1
                elif (self.img[i, j] == 0.0 or j == self.h - 1) and width > 0:
                    stroke_list.append(width)
                    width = 0

        width = 0
        for j in range(self.w):
            for i in range(self.h):
                if self.img[i, j] == 1.0:
                    width += 1
                elif (self.img[i, j] == 0.0 or i == self.w - 1) and width > 0:
                    stroke_list.append(width)
                    width = 0

        stroke_list = np.sort(stroke_list)
        q1 = stroke_list[len(stroke_list) // 4]
        q3 = stroke_list[(len(stroke_list) * 3) // 4]
        iqr = q3 - q1
        mask = (np.array(stroke_list) <= (q3 + 1.5 * iqr)) & (np.array(stroke_list) >= (q1 - 1.5 * iqr))
        return np.round(np.mean(stroke_list[mask]))

    def calc_baselines(self):
        dark_pixel_count = list()
        for i in range(self.h - self.scan_width):
            pixel_count = 0
            for j in range(self.w):
                pixel_count += np.sum([self.img[i + k, j] == 1.0 for k in range(self.scan_width)])
            dark_pixel_count.append(pixel_count)

        beta = 0.95
        gap = self.h // 10

        new_dark_pixel_count = dark_pixel_count.copy()
        for i in range(len(dark_pixel_count)):
            if i == 0:
                new_dark_pixel_count[i] = dark_pixel_count[i]
            else:
                new_dark_pixel_count[i] = beta * new_dark_pixel_count[i - 1] + (1 - beta) * dark_pixel_count[i]
        plt.plot(new_dark_pixel_count, c='black')
        plt.show()

        candidates = []
        flag = True
        min_deg = np.Inf
        for i in range(len(new_dark_pixel_count) - gap * 2):
            first_vector = (-gap, new_dark_pixel_count[i] - new_dark_pixel_count[i + gap])
            second_vector = (gap, new_dark_pixel_count[i + 2 * gap] - new_dark_pixel_count[i + gap])
            deg, positive = calc_angle(first_vector, second_vector)
            if positive:
                flag = False
                if deg <= 90.0:
                    candidates.append(i + gap)
                    break
                else:
                    if deg < min_deg:
                        min_deg = deg
                        candidates.append(i + gap)
            else:
                if not flag:
                    break
        if len(candidates) == 0:
            upper_baseline = 0
        else:
            upper_baseline = candidates[-1]
        candidates.clear()

        new_dark_pixel_count = dark_pixel_count.copy()
        for i in range(len(dark_pixel_count)):
            if i == 0:
                new_dark_pixel_count[i] = dark_pixel_count[len(dark_pixel_count) - i - 1]
            else:
                new_dark_pixel_count[i] = beta * new_dark_pixel_count[i - 1] \
                                          + (1 - beta) * dark_pixel_count[len(dark_pixel_count) - i - 1]
        plt.plot(new_dark_pixel_count, c='black')
        plt.show()

        flag = True
        min_deg = np.Inf
        for i in range(len(new_dark_pixel_count) - 2 * gap):
            first_vector = (-gap, new_dark_pixel_count[i] - new_dark_pixel_count[i + gap])
            second_vector = (gap, new_dark_pixel_count[i + 2 * gap] - new_dark_pixel_count[i + gap])
            deg, positive = calc_angle(first_vector, second_vector)
            if positive:
                flag = False
                if deg <= 90.0:
                    candidates.append(len(dark_pixel_count) - i - gap - 1)
                    break
                else:
                    if deg < min_deg:
                        min_deg = deg
                        candidates.append(len(dark_pixel_count) - i - gap - 1)
            else:
                if not flag:
                    break

        if len(candidates) == 0:
            lower_baseline = self.h - 1
        else:
            lower_baseline = candidates[-1]

        print(gap)
        return upper_baseline, lower_baseline

    def calc_density(self, upper_baseline, lower_baseline):
        cropped = self.img[upper_baseline: lower_baseline, 0:self.w]
        colcnt = np.sum(cropped == 1.0, axis=0)
        pixel_max = np.max(cropped == 1.0, axis=0)
        pixel_min = np.min(cropped == 1.0, axis=0)
        return colcnt, pixel_max, pixel_min

    def detect_segpoints(self, upper_baseline, lower_baseline, colcnt):
        seg = []
        seg1 = []
        seg2 = []

        for i in range(len(colcnt)):
            if colcnt[i] < self.stroke_width:
                seg1.append(i)
        print("Candidates: ", seg1)

        for i in range(len(seg1) - 1):
            if seg1[i + 1] - seg1[i] > self.seperation_threshold:
                seg2.append(seg1[i])

        arr = []
        for i in seg2:
            arr1 = []
            j = upper_baseline
            while j <= lower_baseline:
                if self.img[j, i] == 1.0:
                    arr1.append(1)
                else:
                    arr1.append(0)
                j += 1
            arr.append(arr1)
        print('At arr Seg here: ', seg2)

        ones = []
        for i in arr:
            ones1 = []
            for j in range(len(i)):
                if i[j] == 1:
                    ones1.append(j)
            if len(ones1) == 0:
                ones1.extend([1, 1])
            ones.append(ones1)
        print(ones)

        diffarr = []
        for i in ones:
            diff = i[len(i) - 1] - i[0]
            diffarr.append(diff)
        print('Difference array: ', diffarr)

        for i in range(len(seg2)):
            if diffarr[i] < self.round_threshold:
                seg.append(seg2[i])

        for i in range(len(seg)):
            plt.vlines(seg[i], 0, self.h, colors='black', linewidth=1)

        plt.hlines(upper_baseline, 0, self.w, colors='red', linewidth=1)
        plt.hlines(lower_baseline, 0, self.w, colors='red', linewidth=1)
        plt.imshow(1 - self.img, cmap='gray')
        plt.show()
        return seg

    def segment(self, seg):
        s = 0
        char_list = []
        for i in range(len(seg)):
            if i == 0:
                s = seg[i]
                if s > 15:
                    char_img = self.img[0:, 0:s]
                    cntx = np.count_nonzero(char_img == 1.0)
                    print('count', cntx)
                else:
                    continue
            elif i != (len(seg) - 1):
                if seg[i] - s > 15:
                    char_img = self.img[0:, s:seg[i]]
                    cntx = np.count_nonzero(char_img == 1.0)
                    print('count', cntx)
                    s = seg[i]
                else:
                    continue
            else:
                if seg[i] - s > 15:
                    char_img = self.img[0:, s:seg[i]]
                    cntx = np.count_nonzero(char_img == 1.0)
                    print('count', cntx)
                    char_list.append(char_img)
                    s = seg[i]
                char_img = self.img[0:, seg[len(seg) - 1]:]
                cntx = np.count_nonzero(self.img == 1.0)
                print('count', cntx)

            char_list.append(char_img)

        # fig, axes = plt.subplots(1, len(char_list))
        # for i in range(len(char_list)):
        #     axes[i].imshow(char_list[i], cmap='gray')
        # plt.show()
        return char_list

    def run(self):
        upper_baseline, lower_baseline = self.calc_baselines()
        print(upper_baseline, lower_baseline)

        self.seperation_threshold = (lower_baseline - upper_baseline) // 3
        print(self.seperation_threshold)

        colcnt, pixel_max, pixel_min = self.calc_density(upper_baseline, lower_baseline)

        seg = self.detect_segpoints(upper_baseline, lower_baseline, colcnt)

        char_list = self.segment(seg)
        return char_list
