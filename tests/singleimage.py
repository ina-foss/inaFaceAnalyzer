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

import tensorflow as tf
import unittest
from inaFaceGender.inaFaceGender import GenderImage
from inaFaceGender.face_detector import OcvCnnFacedetector
from inaFaceGender.face_classifier import Resnet50FairFaceGRA

class TestSingleImage(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

   # IMAGE

    def test_image_all_diallo(self):
        gi = GenderImage(face_detector = OcvCnnFacedetector(paddpercent=0.))
        ret = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(ret), 1)
        ret = ret[0]
        self.assertEqual(ret[1], (432, 246, 988, 802))
        self.assertEqual(ret[2], 'f')
        #todo : places = 1 due to changes in rotation procedure
        # same value used in test with micro processing difference
        self.assertAlmostEqual(ret[3], -3.305765917955594, places=1)
        self.assertAlmostEqual(ret[4], 0.99964356, places=4)

    def test_image_all_diallo_multioutput(self):
        gi = GenderImage(face_classifier=Resnet50FairFaceGRA())
        ret = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(ret), 1)
        _, bb, sex, age, sex_decisionf, age_decisionf, face_detect_conf = ret[0]
        self.assertEqual(bb, (421, 234, 997, 810))
        self.assertEqual(sex, 'f')
        self.assertAlmostEqual(age, 25.239, places=2)
        self.assertAlmostEqual(sex_decisionf, -5.258, places=2)
        self.assertAlmostEqual(age_decisionf, 3.023, places=2)
        self.assertAlmostEqual(face_detect_conf, 0.987, places=2)


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
