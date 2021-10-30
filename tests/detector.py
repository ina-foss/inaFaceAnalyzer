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
from numpy.testing import assert_almost_equal
import cv2
from inaFaceGender.face_preprocessing import _squarify_bbox
from inaFaceGender.face_detector import OcvCnnFacedetector, LibFaceDetection


class TestDetector(unittest.TestCase):

    def test_opencv_cnn_detection(self):
        detector = OcvCnnFacedetector(paddpercent=0.)
        img = cv2.imread('./media/Europa21_-_2.jpg')
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret = detector(frame)
        self.assertEqual(len(ret), 1)
        ret, conf = ret[0]
        ret = _squarify_bbox(ret)
        ret = tuple([int(e) for e in ret])
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

    def test_libfacedetection(self):
        detector = LibFaceDetection()
        img = cv2.imread('./media/800px-India_(236650352).jpg')
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = detector(frame)
        self.assertEqual(len(pred), 5)
        ref = [((230.91, 207.73, 455.90, 481.51), 1.0),
            ((496.83, 123.28, 669.31, 325.69), 0.99),
            ((48.75, 34.92, 99.80, 97.77), 0.99),
            ((394.65, 57.10, 446.97, 124.00), 0.99),
            ((229.35, 71.04, 289.13, 147.18), 0.99)]
        for (rbb, rconf), (pbb, pconf) in zip(ref, pred):
            self.assertAlmostEqual(rconf, pconf, places=1)
            assert_almost_equal(list(rbb), list(pbb), decimal=1)
