#!/usr/bin/env python2
# -*- coding:utf-8 -*-
# open_cv: 2.4.13

import numpy as np
import cv2
import cv2.cv as cv

# the Property identifier of  Width, Height, Frame rate,
# Number of frames in the video file
PROP_ID_LIST = [cv.CV_CAP_PROP_FRAME_WIDTH, cv.CV_CAP_PROP_FRAME_HEIGHT,
                  cv.CV_CAP_PROP_FPS, cv.CV_CAP_PROP_FRAME_COUNT]
VIDEO_WITH = 160
VIDEO_HIGH = 120
TOTAL_FRAME_NUM = 40


class VideoLoader(object):
    """get a .avi file, desperate into frame

    Attributes:
        labels: A list of labels of the video will be used in the video classifier
                like [person_id, action, times, uncomp]
        video_path: String of the path and name
        cap: A cv2.VideoCapture of the .avi file

        property of the video of cap:
        cap_w: An integer of video'width  /pic
        cap_h: An integer of video'height  /pic
        frame_rate: An integer of fps(frames per second)
        frame_count: Number of frames in the video file

        property of the frame which video will be divided into,
        image.w: An integer of output frames'width
        image.h: An integer of video'height
        total_image_num: An integer count of the video divided
        frame_matrix: A list where all the frames divided are saved in
        frame_type: string of the color type, 'gray' is default, 'RGB' selected
    """
    def __init__(self, video_path):
        """initiate the class depend on a .avi file"""
        self.video_path = video_path
        self.labels = []
        self.cap = None

        assert self.cap.isOpened()  # make sure the success of reading video

        self._get_label()

        # property of the video of self.cap
        self.cap_w = 0
        self.cap_h = 0
        self.frame_rate = 0
        self.frame_count = 0

        # property of the frame which video will be divided into,
        # this frame is the input of the image processing and machine learning
        self.image_w = self.cap_w
        self.image_h = self.cap_h
        self.total_image_num = self.frame_count
        self.frame_matrix = []

        self.frame_type = 'gray'

    def get_info(self, show_rank=1):
        """
        show the info of the VideoLoader
        :param show_rank: the bigger show_rank is, the more info will be printed
        """
        # TODO(kawawa): Edit here, make a printer

    def print_frames(self):
        """print the frames at rhe last"""

        for frame in self.frame_matrix:
            cv2.imshow("frame", frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    def get_video_loader_info(self):
        """initiate the info of the source video"""
        self.cap_w, self.cap_h, self.frame_rate, self.frame_count = map(self.cap.get, PROP_ID_LIST)
        return self.cap_w, self.cap_h, self.frame_rate, self.frame_count

    def set_frame_property(self, image_w=-1, image_h=-1, total_frame_num=-1):
        """
        set the property of the output frames(self.frame_matrxi), -1 means using the source video property
        :return: None
        """
        if image_w != -1:
            self.image_w = image_w

        if image_h != -1:
            self.image_h = image_h

        if total_frame_num != -1:
            self.total_image_num = total_frame_num

    def set_frame_count(self, frame_count, frame_rate=-1, w=VIDEO_WITH, h=VIDEO_HIGH):
        """just for camera( haven not tested the function, because of the none of device):
        set attributes of the cap(video) for the convenience and efficiency of the video processing.
        :param frame_count: number of frames set in the video file
        :param frame_rate: fps set in the file, -1 means the frame_rate of the source video will be set
        :param w, h: the big w and h will reduce the efficiency of the video process
        """
        if frame_rate is -1:
            frame_rate = int(self.frame_rate)

        map(self.cap.set, [PROP_ID_LIST[0], w], [PROP_ID_LIST[1], h], [PROP_ID_LIST[2], frame_rate],
            [PROP_ID_LIST[3], frame_count])

        self._initiate_video_loader_info()

    def _get_label(self):
        """:return: labels=[person_id, action, times, uncomp]
                    like: [person11, boxing, d4, uncomp]
        """
        path = self.video_path.split('/')
        fileName = path[-1].split(".")[0]
        self.labels = fileName.split("_")

    def get_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def video_to_frame(self, to_gray=True):
        """transform the video to frame,
        :param to_gray: bool if True output the gray value of the frame, else the rgb values.
        :return: all the frames in the cap(video)
        """

        ret, frame = self.cap.read()  # if read successful, ret == True
        frames_counter = 0

        while ret:
            ret, frame = self.cap.read()

            if frame is not None:
                if frames_counter % (self.frame_count/self.total_image_num) < 1:  # to save total_image_num frames
                    if frame is not None:  # at the end of cap.read, return a none frame
                        self.frame_matrix.append(frame)
                frames_counter = frames_counter + 1

        if to_gray:
            gray_frame_matrix = []
            for frame in self.frame_matrix:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # convey frame to gray
                gray_frame_matrix.append(gray_frame)
            self.frame_matrix = gray_frame_matrix
        else:
            self.frame_type = 'RGB'

        type(frame)
        type(self.frame_matrix)

        return self.frame_matrix

    def run(self, frame_count, frame_rate=-1, w=VIDEO_WITH, h=VIDEO_HIGH):
        """the main fun in the VideoLoader
        :return: all the frames in the cap(video)
        """
        return self.video_to_frame()

    def encode_videos(self, fill_path):
        """

        :param fill_path:
        :return:
        """
        # TODO(kawawa): encode all the videos in particular fill


def test(video_path="G:/pycharm/actioncliptest00001.avi", total_frame_num=200):
    """test the VideoLoader"""
    loader = VideoLoader(video_path)
    loader.set_frame_property(total_frame_num=total_frame_num)
    loader.video_to_frame()
    print loader.frame_matrix[0]
    np_array = np.array(loader.frame_matrix)
    print loader.frame_matrix[0]
    print np_array.shape
    print loader.labels
    loader.print_frames()


if __name__ == "__main__":
    # make a test
    test()