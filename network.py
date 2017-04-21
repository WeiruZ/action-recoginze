#! /usr/bin/env python2
# -*- coding:utf-8 -*-

import numpy as np
import matrix_encoder
from matrix_encoder import MatrixEncoder
from nupic.research.spatial_pooler import SpatialPooler
from nupic.research.TP import TP


class HTMNetwork(object):
    """
    Attribute:
    shape: tuple -- set size of the encoder's output, for matrix_encoder,
            it has two int elements.
    """
    def __init__(self,
                 shape=(32, 32),  # tuple -- two element
                 inputDimensions=(1024,),  # tuple two element or int
                 columnDimensions=1024,  # int, tuple is not allowed
                 globalInhibition=1,
                 sp_seed=1960,
                 potentialPct=0.8,
                 synPermConnected=0.10,
                 synPermActiveInc=0.05,
                 synPermInactiveDec=0.0008,
                 maxBoost=2.0,

                 potentialRadius=16,
                 numActiveColumnsPerInhArea=40.0,
                 localAreaDensity=-1.0,
                 stimulusThreshold=0,

                 numberOfCols=1024,  # int
                 cellsPerColumn=16,  # 32 is the official setting
                 tp_seed=1960,
                 newSynapseCount=20,
                 maxSynapsesPerSegment=32,
                 maxSegmentsPerCell=128,
                 initialPerm=0.21,
                 permanenceInc=0.1,
                 permanenceDec=0.0,  # 0.1 is the official setting
                 globalDecay=0,
                 maxAge=0,
                 minThreshold=12,
                 activationThreshold=12,
                 pamLength=1,

                 connectedPerm=0.5,
                 burnIn=2,

                 visible=1):

        # size insurance
        if type(inputDimensions) == int:
            self._assert_fun(shape, (inputDimensions,))
        else:
            self._assert_fun(shape, inputDimensions)
        self._assert_fun((columnDimensions,), (numberOfCols,))

        self.shape = shape

        # the params of the sp
        self.input_dimensions = inputDimensions
        self.column_dimensions = columnDimensions
        self.potential_radius = potentialRadius
        self.numActive_columns_perInhArea = numActiveColumnsPerInhArea
        self.global_inhibition = globalInhibition
        self.syn_perm_active_inc = synPermActiveInc
        self.potential_pct = potentialPct
        self.synPermInactiveDec = synPermInactiveDec
        self.synPermConnected = synPermConnected
        self.sp_seed = sp_seed
        self.localAreaDensity =localAreaDensity
        self.stimulusThreshold = stimulusThreshold
        self.maxBoost = maxBoost

        # the params of the tp
        self.number_of_cols = numberOfCols
        self.cells_per_column =  cellsPerColumn
        self.initial_perm = initialPerm
        self.connected_perm = connectedPerm
        self.min_threshold = minThreshold
        self.new_synapse_count = newSynapseCount
        self.permanence_inc = permanenceInc
        self.permanence_dec = permanenceDec
        self.activation_threshold = activationThreshold
        self.global_decay = globalDecay
        self.burn_in = burnIn
        self.pam_length = pamLength
        self.maxAge = maxAge
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.tp_seed = tp_seed

        self.visible = visible
        self.label = ""

        # network
        self.enc = None
        self.sp = None
        self.tp = None

        self._create_network()

    def set_label(self, label):
        """
        :param label: str -- the tag of the network
        """
        self.label = label

    def get_label(self):
        return self.label

    def _assert_fun(self, param1, param2):
        """
        :param param1, param2: tuple -- contain int type elements.
        make sure two params have a same size.
        """
        product_elements1 = 1
        product_elements2 = 1

        for e in param1:
            product_elements1 = product_elements1 * e
        for i in param2:
            product_elements2 = product_elements2 * i
        assert product_elements1 == product_elements2

    def _check_type(self):
        pass

    def view(self):
        pass

    def _create_network(self, mean=128):
        """
        :param mean: int, the mean of the frame pix value, will be used in BASE_ENCODE.
        """
        # some rulers of creating network
        # the product of the shape's two dimensions is equal to inputDimensions
        # columnDimensions equal to numberOfCols
        self.enc = MatrixEncoder(shape=self.shape, mean=mean)
        self.sp = SpatialPooler(inputDimensions=self.shape[0]*self.shape[1],
                                columnDimensions=self.column_dimensions,
                                potentialRadius=self.potential_radius,
                                numActiveColumnsPerInhArea=self.numActive_columns_perInhArea,
                                globalInhibition=self.global_inhibition,
                                synPermActiveInc=self.syn_perm_active_inc,
                                potentialPct=self.potential_pct,
                                synPermInactiveDec=self.synPermInactiveDec,
                                synPermConnected=self.synPermConnected,
                                seed=self.sp_seed,
                                localAreaDensity=self.localAreaDensity,
                                stimulusThreshold=self.stimulusThreshold,
                                maxBoost=self.maxBoost)
        self.tp = TP(numberOfCols=self.column_dimensions,
                     cellsPerColumn=self.cells_per_column,
                     initialPerm=self.initial_perm,
                     connectedPerm=self.connected_perm,
                     minThreshold=self.min_threshold,
                     newSynapseCount=self.new_synapse_count,
                     permanenceInc=self.permanence_inc,
                     permanenceDec=self.permanence_dec,
                     activationThreshold=self.activation_threshold,
                     globalDecay=self.global_decay,
                     burnIn=self.burn_in,
                     pamLength=self.pam_length,
                     maxSynapsesPerSegment=self.maxSynapsesPerSegment,
                     maxSegmentsPerCell=self.maxSegmentsPerCell,
                     seed=self.tp_seed,
                     maxAge=self.maxAge)

    def _compute(self,a_frame, output, sp_enable_learn, tp_enable_learn):
        """
        the essential proceeding of the network compute,
        the training and prediction is the iteration of it.
        :param a_frame: Array, a frame of the video.
        :param output: np.darray, be used to save the output of the sp.
        """
        matrix = self.enc.encodeIntoArray(a_frame, encoder_model=matrix_encoder.K_MEANS)

        # TODO(kawawa): show the output encoder and sp.
        # image = (np.int16(matrix)-1)*(-255)
        # cv2.imshow("kkk", np.uint8(image))
        # cv2.waitKey(10)
        self.sp.compute(inputVector=matrix, learn=sp_enable_learn, activeArray=output)
        # a = output
        self.tp.compute(bottomUpInput=output, enableLearn=tp_enable_learn, computeInfOutput=None)

    def train(self, frames_matrix, sp_enable_learn=True, tp_enable_learn=True):
        """
        tran the network by a series of frames
        :param frames_matrix: a array of the frames
        :param sp_enable_learn, tp_enable_learn: set the learning model
        """
        output = np.zeros(self.column_dimensions, dtype=int)

        for i in range(len(frames_matrix)):
            self._compute(frames_matrix[i], output, sp_enable_learn, tp_enable_learn)

    def _formatRow(self, x):
        """make a print format"""
        s = ''
        for c in range(len(x)):
            if c > 0 and c % 10 == 0:
                s += ' '
            s += str(x[c])
        s += ' '
        return s

    def predict_detect(self, frames_matrix, sp_enable_learn=False,
                       tp_enable_learn=False):
        """
        get frames, predict the next frame, compare the predicted one with the next input.
        and give a corresponding mark of them.
        :param frames_matrix: a array of the frames
        :param sp_enable_learn, tp_enable_learn: set the learning model
        :return: float -- the corresponding rank of prediction frames and input frames
        """
        output = np.zeros(self.column_dimensions, dtype=int)
        score_list = []

        self._compute(frames_matrix[0], output, sp_enable_learn, tp_enable_learn)
        pre_prediction = self.tp.getPredictedState()

        # view the prediction state
        if self.visible > 1:
            self.tp.printStates(printPrevious=False, printLearnState=False)
            self._formatRow(pre_prediction.max(axis=1).nonzero())

        for i in range(len(frames_matrix))[1:]:
            self._compute(frames_matrix[i], output, sp_enable_learn, tp_enable_learn)
            score = self._give_a_mark(sp_output=output, tp_prediction=pre_prediction)
            score_list.append(score)
            pre_prediction = self.tp.getPredictedState()

            # view the prediction state
            if self.visible > 1:
                self.tp.printStates(printPrevious=False, printLearnState=False)
                self._formatRow(pre_prediction.max(axis=1).nonzero())

        return sum(score_list)

    def getPredictedState(self):
        return self.tp.getPredictedState

    def get_sp_active_cells_index(self, sp_cells_state):
        """
        :return index of active cells/columns in format:
        (array([0, 2, 4], dtype=int64),)
        """
        return sp_cells_state.nonzero()

    def get_tp_active_cells_index(self, tp_cells_state):
        """
        eg:
        the tp_cells _state = [[1, 0],
                               [0, 0],
                               [0, 1]
                               [0, 0]
                               [1, 0]] is a np.ndarray
        :return: index of active columns in format:
        (array([0, 2, 4], dtype=int64),)
        """
        return tp_cells_state.max(axis=1).nonzero()

    def get_tp_active_columns(self, sp_cells_state):
        """
        eg:
        the tp_cells _state = [[1, 0],
                               [0, 0],
                               [0, 1]
                               [0, 0]
                               [1, 0]] is a np.ndarray
        :return: active columns coder [1, 0, 1, 0, 1]
        """
        return sp_cells_state.max(axis=1)

    def _corresponding(self, sp_active_column, tp_active_column):
        """
        compute number of bits where two binary array have the same '1' value.
        sp_active_column and tp_active_column have size 1-d binary array.
        """
        sum = sp_active_column + tp_active_column
        corresponding_elements = sum / 2
        return corresponding_elements.sum()

    def _give_a_mark(self, sp_output, tp_prediction):
        """
        for two frames: next input and the prediction at this time.
        (num of same 1 value bit) /  (num of 1 value bit in sp_output)
        :return: a int between 0-1, 1 means have good prediction
        """
        tp_active_columns =  self.get_tp_active_columns(tp_prediction)
        corresponding_num = self._corresponding(sp_output, tp_active_columns)

        return float(corresponding_num) / float(sum(sp_output))


def test(train_video_path="G:/pycharm/data/gray1/person16_handwaving_d1_uncomp.avi",
         prediction_video_path="G:/pycharm/data/gray1/person16_handwaving_d1_uncomp.avi",
         total_frame_num=20):
    from video_loader import VideoLoader

    def get_video(video_path, total_frame_num):
        loader = VideoLoader(video_path)
        loader.set_frame_property(total_frame_num=total_frame_num)
        frame_matrix = loader.video_to_frame(to_gray=True)

        return frame_matrix

    network = HTMNetwork(visible=1)
    train_frames_matrix = get_video(train_video_path, total_frame_num)
    for i in range(10):
        network.train(frames_matrix=train_frames_matrix)
        print "train iteration {}".format(i)

    predict_frames_matrix = get_video(prediction_video_path, total_frame_num)
    score = network.predict_detect(predict_frames_matrix, False, False)
    print score


if __name__ == "__main__":
    test()