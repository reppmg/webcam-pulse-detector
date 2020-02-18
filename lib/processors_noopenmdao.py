import math
import time

import dlib
import numpy as np
import cv2
import pylab
import os
import sys
import matplotlib.pyplot as plt
import torch
import mobilenet_v1
import torchvision.transforms as transforms
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool


from experiments import transform
from utils.inference import predict_68pts, parse_roi_box_from_landmark, crop_img

PERCENTILE = 0.0001


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10, is_video=False, fps=30):

        self.percentile = PERCENTILE
        self.percentile_time_offset = 10
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.frame_num = 0
        self.cutlow = 1
        self.skipped = 0
        self.base_fps = fps
        self.fps = self.base_fps / self.cutlow
        self.buffer_size = 250
        # self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm_times = []
        self.bpm = 0
        self.bpm_history = []
        self.old_percentile_time = -1
        self.percentile_bpm = 0
        self.percentile_bpm_history = []
        self.percentile_times = []
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

        plt.ion()
        self.fig = plt.figure()
        self.plot = self.fig.add_subplot(111)
        self.fig.show()
        self.plot.set_ylim([45, 180])

        self.is_video = is_video

        self.net = cv2.dnn.readNet("tiny-yolo-azface-fddb_82000.weights", "tiny-yolo-azface-fddb.cfg")
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
        self.model = getattr(mobilenet_v1, 'mobilenet_1')(num_classes=62)

        self.right_brow_point = None
        self.left_brow_point = None

        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        left_x, left_y = self.left_brow_point
        right_x, right_y = self.right_brow_point
        if self.left_brow_point is None:
            return [int(x + w * fh_x - (w * fh_w / 2.0)),
                    int(y + h * fh_y - (h * fh_h / 2.0)),
                    int(w * fh_w),
                    int(h * fh_h)]
        return [
            int(left_x),
            int(left_y),
            int(right_x - left_x),
            int(h * fh_h)
        ]

    # получить среднее значение в прямоугольнике
    def get_subface_means(self, coord):
        x, y, w, h = coord  # грацниы прямогульника
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])  # red component value
        v2 = np.mean(subframe[:, :, 1])  # green
        v3 = np.mean(subframe[:, :, 2])  # blue

        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):
        self.frame_num += 1
        self.times.append(self.current_time())
        display_fps = self.frame_num / (self.current_time() + 0.01)
        self.frame_out = self.frame_in
        # self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
        #                                           cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                        (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Fps: %.2f" % display_fps,
                        (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            self.data_buffer, self.times, self.trained = [], [], False
            self.face_rect = self.detect_faces()

            forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return 1
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
        cv2.putText(
            self.frame_out, "Press 'S' to restart",
            (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                    (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if self.old_percentile_time == -1:
            self.old_percentile_time = self.times[0]
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = self.calc_fps(L)
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 55) & (freqs < 180))

            try:
                pruned = self.fft[idx]
                phase = phase[idx]
            except:
                return

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            if len(pruned) == 0:
                return
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            # r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            # g = alpha * \
            #     self.frame_in[y:y + h, x:x + w, 1] + \
            #     beta * self.gray[y:y + h, x:x + w]
            # b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            # self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
            #                                               g,
            #                                               b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            self.bpms.append(self.bpm)
            # self.ttimes.append(time.time())

            if self.times[-1] - self.old_percentile_time > self.percentile_time_offset:
                sorted_bpms = np.sort(self.bpms)
                trash_amount = int(self.percentile * np.alen(sorted_bpms))
                percentiled_bpms = sorted_bpms[trash_amount:-trash_amount]
                self.percentile_bpm = np.average(percentiled_bpms)
                self.bpms.clear()
                self.percentile_times.append(self.times[-1])
                self.percentile_bpm_history.append(self.percentile_bpm)
                self.old_percentile_time = self.times[-1]
            self.bpm_times.append(self.times[-1])
            self.bpm_history.append(self.bpm)
            if gap:
                text = "(est: %0.1f bpm, perc: %0.1f, wait %0.0f s)" % (self.bpm, self.percentile_bpm, gap)
            else:
                text = "(est: %0.1f bpm, perc: %0.1f)" % (self.bpm, self.percentile_bpm)
                self.draw_bpm_plot()
            tsize = 1
            cv2.putText(self.frame_out, text,
                        (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)

    def parse_roi_box_from_landmark(self, pts):
        """calc roi box from landmark"""
        bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

        llength = math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
        center_x = (bbox[2] + bbox[0]) / 2
        center_y = (bbox[3] + bbox[1]) / 2

        roi_box = [0] * 4
        roi_box[0] = center_x - llength / 2
        roi_box[1] = center_y - llength / 2
        roi_box[2] = roi_box[0] + llength
        roi_box[3] = roi_box[1] + llength

        return roi_box

    def crop_img(self, img, roi_box):
        h, w = img.shape[:2]

        sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
        dh, dw = ey - sy, ex - sx
        if len(img.shape) == 3:
            res = np.zeros((dh, dw, 3), dtype=np.uint8)
        else:
            res = np.zeros((dh, dw), dtype=np.uint8)
        if sx < 0:
            sx, dsx = 0, -sx
        else:
            dsx = 0

        if ex > w:
            ex, dex = w, dw - (ex - w)
        else:
            dex = dw

        if sy < 0:
            sy, dsy = 0, -sy
        else:
            dsy = 0

        if ey > h:
            ey, dey = h, dh - (ey - h)
        else:
            dey = dh

        res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
        return res

    def detect_faces(self):
        # return list(self.face_cascade.detectMultiScale(self.gray,
        #                                                scaleFactor=1.3,
        #                                                minNeighbors=4,
        #                                                minSize=(
        #                                                    50, 50),
        #                                                flags=cv2.CASCADE_SCALE_IMAGE))
        img_ori = self.frame_in
        net = self.net
        layers_names = net.getLayerNames()
        outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(img_ori, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)
        height, width, channels = img_ori.shape
        rects = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    pts = self.face_regressor(img_ori, dlib.rectangle(int(x), int(y), x + w, y + h)).parts()
                    pts = np.array([[pt.x, pt.y] for pt in pts]).T

                    roi_box = parse_roi_box_from_landmark(pts)

                    img = crop_img(img_ori, roi_box)

                    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                    input = self.transform(img).unsqueeze(0)
                    with torch.no_grad():
                        param = self.model(input)
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    pts68 = predict_68pts(param, roi_box)
                    self.left_brow_point = (pts68[0][19], pts68[1][19])
                    self.right_brow_point = (pts68[0][24], pts68[1][24])

                    # for x in range(0, 67):
                    #     cv2.circle(img_ori, (pts68[0][x], pts68[1][x]), 1, (0, 0, 255), -1)
                    cv2.circle(img_ori, (pts68[0][19], pts68[1][19]), 1, (0, 0, 255), -1)
                    cv2.circle(img_ori, (pts68[0][24], pts68[1][24]), 1, (0, 0, 255), -1)

                    return [x, y, w, h]
        return self.face_rect

    def draw_bpm_plot(self):
        if self.frame_num % 10 != 0:
            return
        self.plot.clear()
        times = self.bpm_times[self.buffer_size:]
        bpms = self.bpm_history[self.buffer_size:]
        times, bpms, alg = transform(times, bpms)
        plt.ylim(45, 150)
        self.plot.plot(times, bpms, 'r-')
        self.fig.canvas.draw()

    def calc_fps(self, L):
        if self.is_video:
            return self.base_fps / self.cutlow
        else:
            return float(L) / (self.times[-1] - self.times[0])

    def current_time(self):
        if self.is_video:
            return self.frame_num * (1 / self.fps)
        else:
            return time.time() - self.t0
