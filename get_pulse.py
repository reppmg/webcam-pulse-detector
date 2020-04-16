import time

from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import cv2 as cv
import argparse
import numpy as np
import datetime
import socket
import sys


class getPulseApp(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead. 

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        serial = args.serial
        baud = args.baud
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET,  # Internet
                                      socket.SOCK_DGRAM)  # UDP

        # self.cameras = []
        # self.selected_cam = 0
        # for i in range(3):
        #     camera = Camera(camera=i)  # first camera by default
        #     if camera.valid or not len(self.cameras):
        #         self.cameras.append(camera)
        #     else:
        #         break

        video_file = vars(args)["video"]
        self.is_video = video_file != 0
        self.dataset_name = "%s-dlib-dots" % video_file.split("/")[-1].split(".")[0]  # str(video_file).split("/")[-2]
        self.source = cv.VideoCapture(video_file)
        self.w, self.h = 0, 0
        self.pressed = 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.,
                                          is_video=self.is_video)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.skip = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv,
                             "p": self.pause}

    def pause(self):
        self.skip = not self.skip

    def toggle_cam(self):
        pass
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fmt = "%.4f"
        fn = "results_raw/Webcam-pulse-%s" % (self.dataset_name)
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.bpm_times, self.processor.bpm_history)).T
        np.savetxt(fn + ".csv", data, delimiter=',', fmt=fmt)
        print(np.mean(self.processor.bpm_history))
        print("Writing csv")

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        # state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """

        # handle any key presses
        self.key_handler()

        if self.skip:
            return

        # Get current image frame from the camera
        # frame = self.cameras[self.selected_cam].get_frame()
        ret, frame = self.source.read()
        if self.processor.skipped != self.processor.cutlow - 1:
            self.processor.skipped += 1
            return
        self.processor.skipped = 0
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('frame', gray)

        try:
            self.h, self.w, _c = frame.shape
        except:
            self.write_csv()
            sys.exit(0)

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        res = self.processor.run(0)
        if res == 1 and self.processor.find_faces and self.is_video:
            self.toggle_search()
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        # imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        global st
        global frames
        frames += 1
        if st is None:
            st = time.time()
            return
        fps = frames / (time.time() - st)
        print("FPS = %.2f" % fps)
        if time.time() - st > 5:
            st = None
            frames = 0

    def start(self):
        while self.source.isOpened():
            self.main_loop()
        self.source.release()
        cv.destroyAllWindows()


st = None
frames = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')
    parser.add_argument("-v", "--video", default=0,
                        help="path to the (optional) video file")
    args = parser.parse_args()
    App = getPulseApp(args)
    # while True:
    cv.getBuildInformation()
    App.start()
    App.main_loop()
