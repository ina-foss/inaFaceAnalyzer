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
import numpy as np
from inaFaceGender.inaFaceGender import GenderVideo, GenderImage
from inaFaceGender.face_preprocessing import _norm_bbox, _squarify_bbox
from pandas.util.testing import assert_frame_equal, assert_series_equal
from inaFaceGender.opencv_utils import video_iterator
from inaFaceGender.face_detector import OcvCnnFacedetector
import cv2
from inaFaceGender.face_classifier import Resnet50FairFace, Resnet50FairFaceGRA, Vggface_LSVM_YTF

class TestInaFaceGender(unittest.TestCase):

    def test_image_all_diallo(self):
        gi = GenderImage(face_detector = OcvCnnFacedetector(paddpercent=0.))
        ret = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(ret), 1)
        ret = ret[0]
        self.assertEqual(ret[1], (432, 246, 988, 802))
        self.assertEqual(ret[2], 'f')
        #todo : places = 1 due to changes in rotation procedure
        self.assertAlmostEqual(ret[3], -3.305765917955594, places=1)
        self.assertAlmostEqual(ret[4], 0.99964356, places=4)

    def test_image_knuth(self):
        gi = GenderImage(face_detector = OcvCnnFacedetector(paddpercent=0.))
        ret = gi('./media/20091020222328!KnuthAtOpenContentAlliance.jpg')
        self.assertEqual(len(ret), 1)
        ret = ret[0]
        self.assertEqual(ret[1], (78, 46, 321, 289))
        self.assertEqual(ret[2], 'm')
        #todo : places = 1 due to changes in rotation procedure
        self.assertAlmostEqual(ret[3], 6.621492606578991, places=1)
        self.assertAlmostEqual(ret[4], 0.99995565, places=4)


    def test_video_basic(self):
        gv = GenderVideo(face_detector = OcvCnnFacedetector(paddpercent=0.))
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=25)
        ret.bb = ret.bb.map(str)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv',
                        dtype={'conf': np.float32})
        assert_frame_equal(ret, df, check_less_precise=True)

    # TODO: update with serialized ouput!
    def test_video_res50(self):
        gv = GenderVideo(face_detector = OcvCnnFacedetector(paddpercent=0.), face_classifier=Resnet50FairFace())
        ret = gv('./media/pexels-artem-podrez-5725953.mp4', subsamp_coeff=25)


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



    def test_video_iterator(self):
        src = './media/pexels-artem-podrez-5725953.mp4'
        self.assertEqual(len([e for e in video_iterator(src,start=10, stop=20)]), 11)
        self.assertEqual(len([e for e in video_iterator(src)]), 358)
        self.assertEqual(len([e for e in video_iterator(src, time_unit='ms', start=500, stop=1000)]), 16)


    def test_pred_from_vid_and_bblist(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bb.map(eval))
        _, retdf = gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bb), lbbox)
        self.assertEqual(list(retdf.label), list(df.label))
        assert_series_equal(retdf.decision, df.decision, check_less_precise=True)

    def test_pred_from_vid_and_bblist_res50(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False, face_classifier=Resnet50FairFace())
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # this method read a single face per frame
        df = df.drop_duplicates(subset='frame').reset_index()
        lbbox = list(df.bb.map(eval))
        _, retdf = gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)
        self.assertEqual(len(retdf), len(lbbox))
        self.assertEqual(list(retdf.bb), lbbox)
        # TODO : get test content
        #self.assertEqual(list(retdf.label), list(df.label))
        #assert_series_equal(retdf.decision, df.decision, check_less_precise=True)



    def test_pred_from_vid_and_bblist_boxlist_toolong(self):
        gv = GenderVideo(bbox_scaling=1, squarify=False)
        df = pd.read_csv('./media/pexels-artem-podrez-5725953-notrack-1dectpersec.csv')
        # there will be much more boxes than frames
        lbbox = list(df.bb.map(eval))
        # the processing of these boxes should throw an exception
        with self.assertRaises(AssertionError):
            gv.pred_from_vid_and_bblist('./media/pexels-artem-podrez-5725953.mp4', lbbox, subsamp_coeff=25)

    def test_single_image_single_output_vgg16(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Vggface_LSVM_YTF()
        ret = c(mat)
        self.assertEqual(len(ret), 3)
        feats, label, dec = ret
        self.assertEqual(isinstance(feats, np.ndarray), True, feats)
        self.assertEqual(isinstance(label, str), True, label)
        self.assertEqual(isinstance(dec, float), True, dec)


    def test_single_image_single_output_res50(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFace()
        ret = c(mat)
        self.assertEqual(len(ret), 3)
        feats, label, dec = ret
        self.assertEqual(isinstance(feats, np.ndarray), True)
        self.assertEqual(isinstance(label, str), True, label)
        self.assertEqual(isinstance(dec, np.float32), True, (dec, dec.__class__))

    def test_single_image_multi_output(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFaceGRA()
        ret = c(mat)
        self.assertEqual(len(ret), 3)
        feats, labels, decs = ret
        self.assertEqual(isinstance(feats, np.ndarray), True)
        self.assertEqual(isinstance(labels, tuple), True, labels)
        self.assertEqual(isinstance(decs, tuple), True, decs)
        self.assertEqual(len(labels), 2, labels)
        self.assertEqual(len(decs), 2, decs)

    def test_fairface_age_mapping(self):
        from inaFaceGender.face_classifier import _fairface_agedec2age
        # [(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70+)]
        # simple
        lx = [0, 1, 2, 3]
        ly = [1.5, 6.5, 15, 25]
        np.testing.assert_almost_equal(ly, [_fairface_agedec2age(e) for e in lx])
        # harder - test limits and stuffs around centers...

if __name__ == '__main__':
    unittest.main()
