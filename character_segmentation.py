import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class CharacterSegmentation:
    def __init__(self, img_dir):
        self.path = img_dir
        self.img = Image.open(self.path).convert('L')
        self.img = np.asarray(self.img, dtype='float64')
        self.h, self.w = self.img.shape
        self.preprocess()

    def preprocess(self):
        mask = (self.img / 255) > 0.8
        self.img[mask] = 0.0
        self.img[~mask] = 1.0

    def find_cappoints(self):
        cpoints = []
        dpoints = []
        for i in range(self.w):
            col = self.img[:, i:(i+1)]
            k = col.shape[0]
            while k > 0:
                if col[k-1] == 1.0:
                    dpoints.append((i, k))
                    break
                k -= 1
            for j in range(col.shape[0]):
                if col[j] == 1.0:
                    cpoints.append((i, j))
                    break
        return cpoints, dpoints

    def base_lines(self, upoints, dpoints):
        colu = []
        for i in range(len(upoints)):
            colu.append(upoints[i][1])

        maxyu = max(colu)
        minyu = min(colu)
        avgu = (maxyu + minyu) // 2
        meanu = np.around(np.mean(colu)).astype(int)

        cold = []
        for i in range(len(dpoints)):
            cold.append(dpoints[i][1])

        maxyd = max(cold)
        minyd = min(cold)
        avgd = (maxyu + minyd) // 2
        meand = np.around(np.mean(cold)).astype(int)

        cn = []
        count = 0

        for i in range(self.h):
            for j in range(self.w):
                if self.img[i, j] == 1.0:
                    count += 1
            if count != 0:
                cn.append(count)
                count = 0
        maxindex = cn.index(max(cn))

        plt.hlines(meanu - 6, 0, self.w, colors='black', linewidth=1)
        lb = 0
        if maxindex > meand:
            lb = maxindex
            plt.hlines(maxindex + 6, 0, self.w, colors='black', linewidth=1)
        else:
            lb = meand
            plt.hlines(meand + 6, 0, self.w, colors='black', linewidth=1)

        # plt.imshow(1 - self.img, cmap='gray')
        # plt.show()
        return max(0, meanu - 6), min(lb + 6, self.h - 1)

    def calc_density(self, upper_baseline, lower_baseline):
        cropped = self.img[upper_baseline: lower_baseline, 0:self.w]
        plt.imshow(cropped)
        plt.show()
        colcnt = np.sum(cropped == 1.0, axis=0)
        x = list(range(len(colcnt)))
        plt.plot(colcnt)
        plt.fill_between(x, colcnt, 1, facecolor='blue', alpha=0.5)
        plt.show()
        return colcnt

    def detect_segpoints(self, upper_baseline, lower_baseline, min_pixel_threshold, min_separation_threshold,
                         min_round_letter_threshold, colcnt):
        seg = []
        seg1 = []
        seg2 = []

        for i in range(len(colcnt)):
            if colcnt[i] < min_pixel_threshold:
                seg1.append(i)

        for i in range(len(seg1) - 1):
            if seg1[i + 1] - seg1[i] > min_separation_threshold:
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
                    ones1.append([j])
            if len(ones) == 0:
                ones1.extend([[1], [1]])
            ones.append(ones1)

        diffarr = []
        for i in ones:
            diff = i[len(i) - 1][0] - i[0][0]
            diffarr.append(diff)
        print('Difference array: ', diffarr)

        for i in range(len(seg2)):
            if diffarr[i] < min_round_letter_threshold:
                seg.append(seg2[i])

        for i in range(len(seg)):
            plt.vlines(seg[i], 0, self.h, colors='black', linewidth=1)

        plt.imshow(1 - self.img, cmap='gray')
        plt.show()
        return seg

    def do_segmentation(self, seg):
        s = 0
        char_list = []
        for i in range(len(seg)):
            char_img = None
            if i == 0:
                s = seg[i]
                if s > 15:
                    char_img = self.img[0:, 0:s]
                    cntx = np.count_nonzero(char_img == 1.0)
                    print('count', cntx)
                    # char_list.append(char_img)
                else:
                    continue
            elif i != (len(seg) - 1):
                if seg[i] - s > 15:
                    char_img = self.img[0:, s:seg[i]]
                    cntx = np.count_nonzero(char_img == 1.0)
                    print('count', cntx)
                    # char_list.append(char_img)
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

        fig, axes = plt.subplots(1, len(char_list))
        for i in range(len(char_list)):
            axes[i].imshow(char_list[i], cmap='gray')
        plt.show()
        return char_list

    def segment(self):
        upoints, downpoints = self.find_cappoints()
        meanu, lb = self.base_lines(upoints, downpoints)
        upper_baseline = meanu
        lower_baseline = lb

        colcnt = self.calc_density(upper_baseline, lower_baseline)

        min_pixel_threshold = 6
        min_separation_threshold = 10
        min_round_letter_threshold = 50

        seg = self.detect_segpoints(upper_baseline, lower_baseline, min_pixel_threshold,
                                    min_separation_threshold, min_round_letter_threshold, colcnt)

        char_list = self.do_segmentation(seg)
