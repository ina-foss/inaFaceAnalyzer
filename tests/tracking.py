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
from pandas.testing import assert_frame_equal, assert_series_equal
import tensorflow as tf
import numpy as np
from inaFaceGender.inaFaceGender import GenderTracking, GenderVideo
from inaFaceGender.face_classifier import Resnet50FairFaceGRA, Vggface_LSVM_YTF, Resnet50FairFace
from inaFaceGender.face_detector import OcvCnnFacedetector
from inaFaceGender.face_tracking import Tracker
from inaFaceGender.opencv_utils import imread_rgb
from inaFaceGender.face_utils import _rect_to_tuple


_vid = './media/pexels-artem-podrez-5725953.mp4'

class TestTracking(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_tracker_init(self):
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        bb = [12.2, 70, 666.6666, 200]
        t = Tracker(frame, bb,  None)
        trackbb = _rect_to_tuple(t.t.get_position())
        np.testing.assert_almost_equal(bb, trackbb)

    def test_tracker_updatebb(self):
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        t = Tracker(frame, [100, 100, 200, 200],  None)
        bb = [12.2, 70, 166.6666, 200]
        conf = t.update_from_bb(frame, bb, None)
        trackbb = _rect_to_tuple(t.t.get_position())
        np.testing.assert_almost_equal(bb, trackbb)
        np.testing.assert_almost_equal(2.7552027282149045, conf)
    
    def test_tracking_singleoutput(self):
        gv = GenderTracking(5, face_classifier=Vggface_LSVM_YTF())
        dfpred = gv(_vid, subsamp_coeff=10)
        dfpred.bb = dfpred.bb.map(str)
        dfref = pd.read_csv('./media/pexels-artem-podrez-tracking5-subsamp10-VggFace_LSVM_YTF.csv')
        assert_frame_equal(dfref, dfpred, atol=.01, check_dtype=False)


    # TODO: update with serialized ouput!
    def test_tracking_nofail_multioutput(self):
        gv = GenderTracking(5, face_classifier=Resnet50FairFaceGRA())
        ret = gv(_vid, subsamp_coeff=10)
        # TODO test output
        raise NotImplementedError('testing reference output should be done')

    # compare with and without tracking using non smooth columns only
    def test_trackingVSvideo(self):
        detector = OcvCnnFacedetector(paddpercent=0.)
        for c in [Vggface_LSVM_YTF, Resnet50FairFace, Resnet50FairFaceGRA]:
            classif = c()
            gv = GenderVideo(face_detector = detector, face_classifier = classif)
            gvdf = gv(_vid, subsamp_coeff=30)
            gvdf = gvdf.sort_values(by = ['frame', 'bb']).reset_index(drop=True)
            gt = GenderTracking(1, face_detector = detector, face_classifier = classif)
            gtdf = gt(_vid, subsamp_coeff = 30)
            gtdf = gtdf.sort_values(by = ['frame', 'bb']).reset_index(drop=True)
            for col in gvdf.columns:
                with self.subTest(i=str(c) + col):
                    assert_series_equal(gvdf[col], gtdf[col], check_dtype=False, rtol=0.01)
