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
from inaFaceGender.face_preprocessing import _squarify_bbox
from inaFaceGender.face_detector import OcvCnnFacedetector, LibFaceDetection
from inaFaceGender.opencv_utils import imread_rgb


class TestDetector(unittest.TestCase):

    def test_opencv_cnn_detection(self):
        detector = OcvCnnFacedetector(padd_prct=0.)
        frame = imread_rgb('./media/Europa21_-_2.jpg')
        ret = detector(frame)
        self.assertEqual(len(ret), 1)
        ret, conf = ret[0]
        ret = _squarify_bbox(ret)
        ret = tuple([int(e) for e in ret])
        self.assertEqual(ret, (457, 271, 963, 777))
        self.assertAlmostEqual(conf, 0.99964356)


    def test_opencv_cnn_detection_2(self):
        detector = OcvCnnFacedetector()
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        ret = detector(frame)
        self.assertEqual(len(ret), 5)

        bb, _ = detector.get_closest_face(frame, (200, 200, 500, 450))
        assert_almost_equal(bb, (246.0, 198.2, 439.4, 485.5), decimal = 1)
        bb, _ = detector.get_closest_face(frame, (500, 100, 700, 300), min_iou=.6)
        assert_almost_equal(bb, (501.6, 128.3, 656.5, 328.3), decimal = 1)
        ret = detector.get_closest_face(frame, (700, 0, 800, 200), min_iou=.1)
        self.assertIsNone(ret)

    def test_libfacedetection(self):
        detector = LibFaceDetection()
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        pred = detector(frame)
        self.assertEqual(len(pred), 5)
        ref = [((230.91, 207.73, 455.90, 481.51), 1.0),
            ((496.83, 123.28, 669.31, 325.69), 0.99),
            ((48.75, 34.92, 99.80, 97.77), 0.99),
            ((394.65, 57.10, 446.97, 124.00), 0.99),
            ((229.35, 71.04, 289.13, 147.18), 0.99)]
        for (rbb, rconf), dtc in zip(ref, pred):
            self.assertAlmostEqual(rconf, dtc.conf, places=1)
            assert_almost_equal(list(rbb), list(dtc.bbox), decimal=1)

    def test_libfacedetection_blackpadd(self):
        detector = LibFaceDetection(padd_prct=.1)
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        pred = detector(frame)
        self.assertEqual(len(pred), 5)
        ref = [(229.77, 207.00, 456.69, 480.40),
             (495.26, 129.41, 667.30, 326.08),
             (391.68, 56.72, 452.18, 126.88),
             (228.46, 72.70, 289.66, 147.95),
             (49.14, 33.57, 100.55, 98.27)]
        for rbb, dtc in zip(ref, pred):
            assert_almost_equal(list(rbb), list(dtc.bbox), decimal=1)


    def test_minpx(self):
        detector = LibFaceDetection(min_size_px = 80)
        frame = imread_rgb('./media/800px-India_(236650352).jpg')
        pred = detector(frame)
        self.assertEqual(len(pred), 2)
        ref = [((230.91, 207.73, 455.90, 481.51), 1.0),
            ((496.83, 123.28, 669.31, 325.69), 0.99)]
        for (rbb, rconf), dtc in zip(ref, pred):
            self.assertAlmostEqual(rconf, dtc.conf, places=1)
            assert_almost_equal(list(rbb), list(dtc.bbox), decimal=1)
