#! /usr/bin/env python2
# -*- coding:utf-8 -*-

import video_loader
from video_loader import VideoLoader

class BaseClassifier(object):
    """
    """

    def classify(self, video_path):
        """
        :return:
        """
        pass


class PredictionClassifier(BaseClassifier):
    """
    easily classify video by the prediction of particular htm,
    the each htm in the htm_networks has trained
    by a series of video belong to particular class
    input the frames_matrix into all the htms, prediction and check,
    the most corresponding htm will specify the class video belong to.

    Attributions:
    htm_networks: a list, contain many htm_networks.
    labels: a list, contain all the video class labels,
             htm_networks and label a corresponding order.
    frames_matrix: a np.ndarray -- save frames of one video.
    """
    def __init__(self, htm_networks, labels, total_frame_num):
        """
        """
        assert len(htm_networks) == len(labels)

        self.htm_networks = htm_networks
        self.labels = labels
        self.total_frame_num = total_frame_num
        self.len = len(labels)
        self.to_gray = True
        self.frames_matrix = None

        self.score_dict = {}

    def set_gray(self, to_gray):
        """
        :param to_gray: bool -- True: transform to gray pix
        """
        self.to_gray =to_gray

    def video_to_frame(self, video_path, total_frame_num):
        loader = VideoLoader(video_path)
        loader.set_frame_property(total_frame_num=total_frame_num)
        frames_matrix = loader.video_to_frame(to_gray=self.to_gray)

        return frames_matrix

    def score(self, htm_network, frames_matrix, sp_enable_learn=False, tp_enable_learn=False):
        """
        :param htm_network:
        :return: a score list.
        """

        score = htm_network.predict_detect(frames_matrix, sp_enable_learn,
                                   tp_enable_learn)
        return score

    def scores(self, frames_matrix):
        classify_criterion = []
        for i in range(self.len):
            score = self.score(self.htm_networks[i], frames_matrix)
            classify_criterion.append(score)

        self.score_dict = dict(zip(self.labels, classify_criterion))
        return classify_criterion

    def reset_frames_matrix(self):
        self.frames_matrix = None

    def classify(self, video_path):
        """
        :return:
        """
        if self.frames_matrix == None:
            self.frames_matrix = self.video_to_frame(video_path, total_frame_num=self.total_frame_num)

        classify_criterion = self.scores(self.frames_matrix)
        max_score_index = classify_criterion.index(max(classify_criterion))
        label = self.labels[max_score_index]

        self.reset_frames_matrix()

        return label, classify_criterion