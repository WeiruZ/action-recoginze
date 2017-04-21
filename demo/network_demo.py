#! /usr/bin/env python2
# -*- coding:utf-8 -*-


import sys
import numpy as np
sys.path.append("..")

from video_loader import VideoLoader
from matrix_encoder import MatrixEncoder
from nupic.research.spatial_pooler import SpatialPooler
from nupic.research.TP import TP

def get_video(video_path="G:/pycharm/data/gray1/person16_handclapping_d1_uncomp.avi", total_frame_num=200):
    loader = VideoLoader(video_path)
    loader.set_frame_property(total_frame_num=total_frame_num)
    frame_matrix = loader.video_to_frame(to_gray=True)

    return frame_matrix

def create_network():
    enc = MatrixEncoder((64,64))
    sp = SpatialPooler(inputDimensions=4096,columnDimensions=1024)
    tp = TP(numberOfCols=1024)

    return enc, sp, tp

def frame_to_vector(frame):
    mean = 128

    return np.array(frame)/mean

def run_network():
    frame_matrix = get_video()
    vector = frame_to_vector(frame_matrix)
    a = len(vector)
    enc, sp, tp = create_network()
    output = np.zeros(1024, dtype=int)

    for i in range(len(vector)):
        matrix = enc.encodeIntoArray(vector[i])

        sp.compute(matrix, learn=True, activeArray=output)
        tp.compute(output,True,computeInfOutput=True)

        # print output

run_network()