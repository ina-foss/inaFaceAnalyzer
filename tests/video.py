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
import tensorflow as tf
from inaFaceGender.inaFaceGender import GenderVideo, VideoPrecomputedDetection
from inaFaceGender.face_classifier import Resnet50FairFace, Resnet50FairFaceGRA, Vggface_LSVM_YTF
from inaFaceGender.face_detector import LibFaceDetection, PrecomputedDetector

_vid = './media/pexels-artem-podrez-5725953.mp4'

class TestVideo(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

    # VIDEO

    # this test execution time is too long
    # def test_video_simple(self):
    #     gv = GenderVideo(face_classifier = Vggface_LSVM_YTF())
    #     ret = gv(_vid)
    #     ret.bbox = ret.bbox.map(str)
    #     df = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv')
    #     assert_frame_equal(ret, df, atol=.01, check_dtype=False)


    def test_video_subsamp(self):
        gv = GenderVideo(face_classifier = Vggface_LSVM_YTF())
        ret = gv(_vid, fps=1)
        ret.bbox = ret.bbox.map(str)
        refdf = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv')
        refdf = refdf[(refdf.frame % 30) == 0].reset_index(drop=True)
        assert_frame_equal(refdf, ret, rtol=.01, check_dtype=False)

    # TODO: update with serialized ouput!
    def test_video_res50(self):
        gv = GenderVideo(face_classifier=Resnet50FairFace())
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', fps=1)
        raise NotImplementedError('test should be improved')

    def test_video_libfacedetection(self):
         gv = GenderVideo(face_detector=LibFaceDetection())
         ret = gv('./media/pexels-artem-podrez-5725953.mp4', fps=1)
         raise NotImplementedError('test should be improved')


    def test_videocall_multioutput(self):
        gv = GenderVideo(face_classifier=Resnet50FairFaceGRA())
        preddf = gv('./media/pexels-artem-podrez-5725953.mp4', fps=1)
        refdf = pd.read_csv('./media/pexels-artem-podrez-subsamp30-Resnet50FFGRA.csv')
        refdf.bbox = refdf.bbox.map(eval)
        assert_frame_equal(refdf, preddf, rtol=.01, check_dtype=False)


    def test_pred_from_vid_and_bblist(self):
        gv = VideoPrecomputedDetection(bbox_scaling=1, squarify_bbox=False, face_classifier = Vggface_LSVM_YTF())


        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notracking.csv',
                        dtype={'face_detect_conf': np.float32})
        df = df[(df.frame % 25) == 0].reset_index(drop=True)


        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bbox.map(eval))
        retdf = gv('./media/pexels-artem-podrez-5725953.mp4', lbbox, fps=30/25.)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bbox), lbbox)
        self.assertEqual(list(retdf.sex_label), list(df.sex_label))
        assert_series_equal(retdf.sex_decfunc, df.sex_decfunc, rtol=.01)

    def test_pred_from_vid_and_bblist_multioutput(self):
        gv = VideoPrecomputedDetection(bbox_scaling=1, squarify_bbox=False, face_classifier=Resnet50FairFaceGRA())
        df = pd.read_csv('./media/pexels-artem-podrez-subsamp30-Resnet50FFGRA.csv')
        # this trick keeps only single face per frame
        df = df.drop_duplicates(subset='frame').reset_index(drop=True)
        df.bbox = df.bbox.map(eval)
        lbbox = list(df.bbox)
        df = df.drop(['detect_conf'], axis=1)
        retdf = gv('./media/pexels-artem-podrez-5725953.mp4', lbbox, fps=1)

        assert_frame_equal(df, retdf.drop(['detect_conf'], axis=1), rtol=.01, check_dtype=False)


    def test_pred_from_vid_and_bblist_res50(self):
        gv = VideoPrecomputedDetection(bbox_scaling=1, squarify_bbox=False, face_classifier=Resnet50FairFace())
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bbox.map(eval))
        retdf = gv('./media/pexels-artem-podrez-5725953.mp4', lbbox, fps=30./25)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bbox), lbbox)
        # TODO : get test content
        #self.assertEqual(list(retdf.label), list(df.label))
        #assert_series_equal(retdf.decision, df.decision, rtol=.01)
        raise NotImplementedError('reference csv should be generated for this test')




    def test_pred_from_vid_and_bblist_boxlist_toolong(self):
        gv = VideoPrecomputedDetection()
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # there will be much more boxes than frames
        lbbox = list(df.bbox.map(eval))
        # the processing of these boxes should throw an exception
        with self.assertRaises(AssertionError):
            gv('./media/pexels-artem-podrez-5725953.mp4', lbbox, fps=30./25)

    def test_vid_nofaces(self):
        gv = GenderVideo(face_classifier=Resnet50FairFaceGRA(), face_detector=PrecomputedDetector([[]]))
        df = gv(_vid, fps=1)
        self.assertEqual(len(df), 0)
        self.assertEqual(len(df.columns), 7, df.columns)
