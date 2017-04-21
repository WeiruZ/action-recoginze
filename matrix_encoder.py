from nupic.encoders.base import Encoder
import numpy as np
import cv2

K_MEANS = 1
BASE_ENCODE=2

class MatrixEncoder(Encoder):
    """
    Attribute:
    shape: A tuple, define the shape ot the SDRs matrix
    mean: A int is the mean of the pix value, will be used in binary transform
    """

    def __init__(self, shape, mean=128):
        self.shape = shape
        self.mean = mean

    def encodeIntoArray(self, input, encoder_model=K_MEANS):
        """
        :param input:  a np.ndarray type -- a frame
        :param encoder_model: set the way of encode, BASE_ENCODER or K_MEANS
        :return:  self.shape 0-1 matrix
        """
        if input is not None and not isinstance(input, np.ndarray):
            raise TypeError(
                "Expected a np.ndarray input but got input of type %s" % type(input)
            )
        input_one_d = input.reshape(1, -1)[0]
        if encoder_model == 2:
            binary_one_d = self.easy_to_binary(input_one_d)
        elif encoder_model == 1:
            binary_one_d = self.k_means_to_binary(input_one_d)

        binary_matrix = binary_one_d.reshape(input.shape)
        shape_binary_matrix = self._reset_size(binary_matrix)

        # make a xor process.
        shape_binary_matrix = MatrixEncoder.xor_trans(shape_binary_matrix, 1)
        return shape_binary_matrix

    def easy_to_binary(self, frame):
        return np.array(frame) / self.mean

    def k_means_to_binary(self, a_frame, K=2):
        """
        at early stage, K can only set to 2
        :param a_frame: a np.ndarray type -- a frame
        :param K:
        :return:
        """
        recolor = self.k_means(a_frame, K)
        max = np.max(recolor)
        return recolor / max

    def k_means(self, a_frame, K=2):
        """
        :param a_frame:
        :param K:
        :return: np.ndarray draw the frame use K color's centers
        """
        i = 0
        Z = a_frame.reshape((-1, 1))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((a_frame.shape))

        return res2

    def _reset_size(self, matrix):
        """
        Nearest Neighbour Resampling
        :param matrix:
        :return:
        """
        if matrix is None and not isinstance(matrix, np.ndarray):
            raise TypeError(
                "Expected a np.ndarray input but got input of type %s" % type(matrix)
            )

        shape = matrix.shape

        x_ratio = float(self.shape[0]) / shape[0]
        y_ratio = float(self.shape[1]) / shape[1]

        d_x_list = np.zeros(self.shape[0], dtype=int)  # save the src matrix pix's x_coordinate which will be write in the new matrix in order.
        d_y_list = np.zeros(self.shape[1], dtype=int)  # same to the ahead

        # compute the the x and y coordinate of d_pix in the src matrix
        for x in range(shape[0]):
            d_x = int(x * x_ratio)
            d_x_list[d_x] = int(x)

        for y in range(shape[1]):
            d_y = int(y * y_ratio)
            d_y_list[d_y] = int(y)

        # get the index matrix to acquire the destined elements in the src matrix
        T_x = np.ones((self.shape[0], 1), dtype=int)
        x_index_matrix = T_x * d_x_list
        T_y = np.ones((self.shape[1], 1), dtype=int)
        y_index_matrix = T_y * d_y_list
        new_matrix = matrix[x_index_matrix.transpose(), y_index_matrix]

        return new_matrix

    def get_shape(self):
        return self.shape

    @staticmethod
    def xor_trans(value1, value2):
        # TODO(kawawa): implement a wide applicability method
        return value1 ^ value2


def test():
    enc = MatrixEncoder((4, 4))

if __name__ == "__main__":
    enc = MatrixEncoder((4,4))
    print enc.encodeIntoArray(np.array([[9,3,4,10,10],[0,0,0,0,10],[0,2,0,3,1],[1,2,0,0,6]]))
