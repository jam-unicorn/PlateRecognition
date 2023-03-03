import cv2
import numpy as np
from numpy.linalg import norm
from hyperlpr import HyperLPR_PlateRecogntion
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse


MAX_WIDTH = 1000
MIN_AREA = 2000
SZ = 20
PROVINCE_START = 1000
PROVINCES = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]


def img_read(file):
    return cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)


def img_find_contours(img_contours):
    contours, hierarchy = cv2.findContours(
        img_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    car_contours = []
    boxPoints = []
    for cnt in contours:
        ant = cv2.minAreaRect(cnt)
        width, height = ant[1]
        if width < height:
            width, height = height, width
        ration = width / height
        if 2 < ration < 5.5:
            car_contours.append(ant)
            box = cv2.boxPoints(ant)
            boxPoints.append(box)
    return car_contours, boxPoints


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def img_transform(car_contours, oldimg, pic_width, pic_hight):
    car_imgs = []
    for car_rect in car_contours:
        if -1 < car_rect[2] < 1:
            angle = 1
        else:
            angle = car_rect[2]
        car_rect = (car_rect[0], (car_rect[1][0] +
                    5, car_rect[1][1] + 5), angle)
        box = cv2.boxPoints(car_rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        if left_point[1] <= right_point[1]:
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            car_img = dst[int(left_point[1]):int(heigth_point[1]), int(
                left_point[0]):int(new_right_point[0])]
            car_imgs.append(car_img)
        elif left_point[1] > right_point[1]:
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            car_img = dst[int(right_point[1]):int(heigth_point[1]), int(
                new_left_point[0]):int(right_point[0])]
            car_imgs.append(car_img)
    return car_imgs


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    row_num_limit = 21
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


def img_color(card_imgs):
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = None
        try:
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        except:
            print("矫正矩形出错, 转换失败")
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num
        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:
                    yello += 1
                elif 35 < H <= 99 and S > 34:
                    green += 1
                elif 99 < H <= 124 and S > 34:
                    blue += 1
                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"
        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124
        elif black + white >= card_img_count * 0.7:
            color = "bw"
        colors.append(color)
        card_imgs[card_index] = card_img
        if limit1 == 0:
            continue
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        if color == "green":
            card_imgs[card_index] = card_img
        else:
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                yl - (
                    yh - yl) // 4:yh,
                xl:xr]
        if need_accurate:
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(
                card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        if color == "green":
            card_imgs[card_index] = card_img
        else:
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                yl - (
                    yh - yl) // 4:yh,
                xl:xr]
    return colors, card_imgs


def find_waves(threshold, histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
                 for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)
    return np.float32(samples)


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def train_svm(self):
        self.model = SVM(C=1, gamma=0.5)
        self.modelchinese = SVM(C=1, gamma=0.5)
        self.model.load("models/svm.dat")
        self.modelchinese.load("models/svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        img = car_pic_file
        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(
                pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        blur = 5
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        ret, img_thresh = cv2.threshold(
            img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_edge = cv2.Canny(img_thresh, 100, 200)
        Matrix = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        return img_edge2, oldimg

    def img_color_contours(self, img_contours, oldimg, add_box_point=False):
        pic_hight, pic_width = img_contours.shape[:2]
        card_contours, boxPoints = img_find_contours(img_contours)
        card_imgs = img_transform(
            card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_color(card_imgs)
        predict_result = []
        predict_str = ""
        roi = None
        card_color = None
        box_point = None
        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                try:
                    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                except:
                    pass
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(
                    gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    continue
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                row_num, col_num = gray_img.shape[:2]
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5
                wave_peaks = find_waves(y_threshold, y_histogram)
                if len(wave_peaks) <= 6:
                    continue
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)
                cur_dis = 0
                for wi, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if wi > 0:
                    wave = (wave_peaks[0][0], wave_peaks[wi][1])
                    wave_peaks = wave_peaks[wi + 1:]
                    wave_peaks.insert(0, wave)
                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)
                if len(wave_peaks) <= 6:
                    continue
                part_cards = seperate_card(gray_img, wave_peaks)
                for pi, part_card in enumerate(part_cards):
                    if np.mean(part_card) < 255 / 5:
                        continue
                    part_card_old = part_card
                    w = abs(part_card.shape[1] - SZ) // 2
                    part_card = cv2.copyMakeBorder(
                        part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(
                        part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    part_card = preprocess_hog([part_card])
                    if pi == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = PROVINCES[int(
                            resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(np.int32(resp[0]))
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)
                roi = card_img
                card_color = color
                box_point = boxPoints[i]
                break
        if (add_box_point == True):
            return predict_str, roi, card_color, box_point
        return predict_str, roi, card_color


app = FastAPI()


@app.post('/')
async def run(file: UploadFile):
    predictor = CardPredictor()
    predictor.train_svm()
    img_bgr = img_read(file.file)
    first_img, oldimg = predictor.img_first_pre(img_bgr)
    r_c, roi_c, color_c = predictor.img_color_contours(first_img, oldimg)
    try:
        Plate = HyperLPR_PlateRecogntion(img_bgr)
        r_c = Plate[0][0]
    except:
        pass
    if r_c:
        return JSONResponse(status_code=200, content={'result': r_c})
    else:
        return JSONResponse(status_code=500, content={})
