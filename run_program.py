import os
from video_loader import VideoLoader
from network import HTMNetwork
from classifier import PredictionClassifier
import cPickle as pickle
import numpy as np

TOTAL_FRAME_NUM = 40
gray_labels = ["boxing", "handclapping", "handwaving",
                    "jogging", "running", "walking"]
gray_path = "G:/actionRecognition/DataSets/gray"
TRAIN_SIZE = 3

def get_video(video_path, total_frame_num):
    loader = VideoLoader(video_path)
    loader.set_frame_property(total_frame_num=total_frame_num)
    frame_matrix = loader.video_to_frame(to_gray=True)

    return frame_matrix

def get_batch_video_name(label, path):
    """"""
    name_list = os.listdir(path+"/"+label)

    return name_list

def train(network, path, label, name_list, train_size):
    c = 0
    for i in range(train_size):
        frames_matrix = get_video(path+"/"+str(label)+"/"+name_list[i], total_frame_num=TOTAL_FRAME_NUM)
        network.train(frames_matrix)
        print label, c
        c += 1

    jar = open("trained_network/"+label+"_network.pkl", "wb")
    pickle.dump(network, jar)
    jar.close()

def train_networks(label_list, data_path, train_size):
    network_list = []
    for i in range(len(label_list)):
        network = HTMNetwork()
        network.set_label(label_list[i])
        network_list.append(network)
        name_list = get_batch_video_name(label_list[i], data_path)
        train(network_list[i], data_path, label_list[i], name_list, train_size=train_size)

    return network_list

def load_networks():
    network_list = []
    label_list = []
    network_name = os.listdir("trained_network")
    for label in network_name:
        pkl_file = open("trained_network/"+label)
        network = pickle.load(pkl_file)
        network_list.append(network)
        label_list.append(network.get_label())
        pkl_file.close()

    return network_list, label_list

def train_all_data(label_list=gray_labels, data_path=gray_path, train_size=TRAIN_SIZE):
    network_list = train_networks(gray_labels, gray_path, train_size)
    return network_list

def classify(network_list, label_list, video_path):
    classifier = PredictionClassifier(network_list, label_list, total_frame_num=TOTAL_FRAME_NUM)
    label, classify_criterion = classifier.classify(video_path)
    return label, classify_criterion

def test(network_list, label_list, path, test_size):
    bingo_num = np.zeros(len(label_list))
    classify_criterion_list = [[]] * len(label_list)

    for i in range(len(label_list)):
        bingo = 0
        name_list = get_batch_video_name(label_list[i], path)
        label = label_list[i]
        p = 0

        for c in range(test_size):
            print label, p
            p += 1
            index = -1 - c
            pre_label, classify_criterion = classify(network_list, label_list,
                                 path+"/"+str(label)+"/"+name_list[index])
            print "pre_label:" + pre_label
            print classify_criterion
            classify_criterion_list[i].append(classify_criterion)
            if pre_label == label:
                bingo += 1

        bingo_num[i] = float(bingo)

    return np.array(bingo_num)/float(test_size)

def main1():
    train_all_data(train_size=90)

    network_list, label_list = load_networks()
    video_path = gray_path + "/handwaving/person01_handwaving_d1_uncomp.avi"
    print classify(network_list, label_list, video_path)

def main2():
    network_list, label_list = load_networks()
    print test(network_list, label_list, gray_path, test_size=10)

main2()