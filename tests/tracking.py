#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import unittest
import pandas as pd
from inaFaceGender.inaFaceGender import GenderTracking
from inaFaceGender.face_classifier import Resnet50FairFaceGRA, Vggface_LSVM_YTF
import tensorflow as tf

class TestTracking(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

    # TODO: update with serialized ouput!
    def test_tracking_nofail_singleoutput(self):
        gv = GenderTracking(face_classifier=Vggface_LSVM_YTF(), detection_period=5)
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=10)
        # TODO test output

    # TODO: update with serialized ouput!
    def test_tracking_nofail_multioutput(self):
        gv = GenderTracking(face_classifier=Resnet50FairFaceGRA(), detection_period=5)
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=10)
        # TODO test output

## TODO compare to gender video on non smooth stuffs
