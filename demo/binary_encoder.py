#! /usr/bin/env python2
# -*- coding:utf-8 -*-

import sys
import numpy as np
from PIL import Image
import cv2

sys.path.append("..")

from video_loader import VideoLoader

# def imageToVector(image):
#     '''
#     Returns a bit vector representation (list of ints) of a PIL image.
#     '''
#     # Convert the image to black and white
#     image = image.convert('1',dither=Image.NONE)
#     # Pull out the data, turn that into a list, then a numpy array,
#     # then convert from 0 255 space to binary with a threshold.
#     # Finally cast the values into a type CPP likes
#     vector = (numpy.array(list(image.getdata())) < 100).astype('uint32')
#
#     return vector
#
# def imagesToVectors(images):
#     vectors = [imageToVector(image) for image in images]
#     return vectors

def frame_to_vector(frame):
    mean = 128

    return np.array(frame)/mean

def get_video(video_path=
              "G:/pycharm/data/Hollywood2_format/AVIClips/actionclipautoautotrain00078.avi",
              total_frame_num=200):
    loader = VideoLoader(video_path)
    loader.set_frame_property(total_frame_num=total_frame_num)
    frame_matrix = loader.video_to_frame()

    return frame_matrix

images = get_video()
vector = frame_to_vector(images)
print vector.shape
for i in range(199):
    cv2.imshow("im", vector[i]*223)
    cv2.waitKey(20)