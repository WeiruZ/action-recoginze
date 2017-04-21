#! /usr/bin/env python2
# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2

sys.path.append("..")

from video_loader import VideoLoader

path_gray = "G:/pycharm/walking1.avi"
path_rgb = "G:/pycharm/data/gray1/person11_jogging_d4_uncomp.avi"

def get_video(video_path=
             path_gray , total_frame_num=200):
    loader = VideoLoader(video_path)
    loader.set_frame_property(total_frame_num=total_frame_num)
    frame_matrix = loader.video_to_frame(to_gray=True)

    return frame_matrix


def cluster(frame_matrix):
    new_frame_matrix = []
    i = 0
    for frame in frame_matrix:
        print "reader {} frame".format(i)
        i += 1
        Z = frame.reshape((-1, 1))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2

        ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((frame.shape))

        new_frame_matrix.append(res2)
        cv2.imshow('res2', res2)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def test():
    frame_matrix = get_video()
    cluster(frame_matrix)


if __name__ == "__main__":
    test()