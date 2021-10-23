#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019 Ina (David Doukhan - http://www.ina.fr/)

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
import numpy as np
import cv2
import tensorflow as tf
from inaFaceGender.inaFaceGender import GenderVideo
from inaFaceGender.face_preprocessing import _norm_bbox, _squarify_bbox
from inaFaceGender.face_detector import OcvCnnFacedetector
from inaFaceGender.face_classifier import Resnet50FairFace, Resnet50FairFaceGRA, Vggface_LSVM_YTF

_vid = './media/pexels-artem-podrez-5725953.mp4'

class TestIFG(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

    # VIDEO

    def test_video_simple(self):
        gv = GenderVideo()
        ret = gv(_vid)
        ret.bbox = ret.bbox.map(str)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv')
        assert_frame_equal(ret, df, atol=.01, check_dtype=False)


    def test_video_subsamp(self):
        gv = GenderVideo()
        ret = gv(_vid, subsamp_coeff=30)
        ret.bbox = ret.bbox.map(str)
        refdf = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv')
        refdf = refdf[(refdf.frame % 30) == 0].reset_index(drop=True)
        assert_frame_equal(refdf, ret, rtol=.01, check_dtype=False)

    # TODO: update with serialized ouput!
    def test_video_res50(self):
        gv = GenderVideo(face_classifier=Resnet50FairFace())
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=25)
        raise NotImplementedError('test should be improved')

    def test_videocall_multioutput(self):
        gv = GenderVideo(face_classifier=Resnet50FairFaceGRA())
        preddf = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=30)
        refdf = pd.read_csv('./media/pexels-artem-podrez-subsamp30-Resnet50FFGRA.csv')
        refdf.bbox = refdf.bbox.map(eval)
        assert_frame_equal(refdf, preddf, rtol=.01, check_dtype=False)

    # DETECTOR

    def test_opencv_cnn_detection(self):
        detector = OcvCnnFacedetector(paddpercent=0.)
        img = cv2.imread('./media/Europa21_-_2.jpg')
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret = detector(frame)
        self.assertEqual(len(ret), 1)
        ret, conf = ret[0]
        ret = _squarify_bbox(ret)
        ret = _norm_bbox(ret, img.shape[1], img.shape[0])
        self.assertEqual(ret, (457, 271, 963, 777))
        self.assertAlmostEqual(conf, 0.99964356)


    def test_opencv_cnn_detection_2(self):
        detector = OcvCnnFacedetector()
        img = cv2.imread('./media/800px-India_(236650352).jpg')
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret = detector(frame)
        self.assertEqual(len(ret), 5)

        bb, _ = detector.get_closest_face(frame, (200, 200, 500, 450))
        self.assertAlmostEqual(bb, [246.0497784614563, 198.21387606859207, 439.4639492034912, 485.5670797228813])
        bb, _ = detector.get_closest_face(frame, (500, 100, 700, 300), min_iou=.6)
        self.assertAlmostEqual(bb, [501.6525077819824, 128.37764537334442, 656.5784645080566, 328.3189299106598])
        ret = detector.get_closest_face(frame, (700, 0, 800, 200), min_iou=.1)
        self.assertIsNone(ret)

    # VIDEO

    def test_pred_from_vid_and_bblist(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False)


        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv',
                        dtype={'face_detect_conf': np.float32})
        df = df[(df.frame % 25) == 0].reset_index(drop=True)


        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bbox.map(eval))
        _, retdf = gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bbox), lbbox)
        self.assertEqual(list(retdf.sex_label), list(df.sex_label))
        assert_series_equal(retdf.sex_decfunc, df.sex_decfunc, rtol=.01)

    def test_pred_from_vid_and_bblist_multioutput(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False, face_classifier=Resnet50FairFaceGRA())
        df = pd.read_csv('./media/pexels-artem-podrez-subsamp30-Resnet50FFGRA.csv')
        # this trick keeps only single face per frame
        df = df.drop_duplicates(subset='frame').reset_index(drop=True)
        df.bbox = df.bbox.map(eval)
        lbbox = list(df.bbox)
        df = df.drop(['face_detect_conf', 'frame'], axis=1)
        _, retdf = gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=30)

        assert_frame_equal(df, retdf, rtol=.01, check_dtype=False)


    def test_pred_from_vid_and_bblist_res50(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False, face_classifier=Resnet50FairFace())
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bbox.map(eval))
        _, retdf = gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bbox), lbbox)
        # TODO : get test content
        #self.assertEqual(list(retdf.label), list(df.label))
        #assert_series_equal(retdf.decision, df.decision, rtol=.01)
        raise NotImplementedError('reference csv should be generated for this test')




    def test_pred_from_vid_and_bblist_boxlist_toolong(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # there will be much more boxes than frames
        lbbox = list(df.bbox.map(eval))
        # the processing of these boxes should throw an exception
        with self.assertRaises(AssertionError):
            gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)

    def test_vid_nofaces(self):
        gv = GenderVideo(face_classifier=Resnet50FairFaceGRA(), face_detector=lambda x: [])
        df = gv(_vid, subsamp_coeff=30)
        self.assertEqual(len(df), 0)
        self.assertEqual(len(df.columns), 7, df.columns)
