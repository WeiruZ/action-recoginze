#! /usr/bin/env python2
# -*- coding:utf-8 -*-

import sys
sys.path.append("..")
from network import HTMNetwork
import pickle

network = HTMNetwork()

jar = open("network.pkl", "wb")
pickle.dump(network, jar)
jar.close()