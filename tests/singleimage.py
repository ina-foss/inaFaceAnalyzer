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
        df = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(df), 1)
        self.assertEqual(df.bbox[0], (432, 246, 988, 802))
        self.assertEqual(df.sex_label[0], 'f')
        self.assertAlmostEqual(df.sex_decfunc[0], -3.323815, places=5)
        self.assertAlmostEqual(df.face_detect_conf[0], 0.99964356, places=4)

    def test_image_all_diallo_multioutput(self):
        gi = GenderImage(face_classifier=Resnet50FairFaceGRA())
        df = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(df), 1)
        e = next(df.itertuples(index=False, name = 'useless'))
        self.assertEqual(e.bbox, (421, 234, 997, 810))
        self.assertEqual(e.sex_label, 'f')
        self.assertAlmostEqual(e.age_label, 25.239, places=2)
        self.assertAlmostEqual(e.sex_decfunc, -5.258, places=2)
        self.assertAlmostEqual(e.age_decfunc, 3.023, places=2)
        self.assertAlmostEqual(e.face_detect_conf, 0.987, places=2)


    def test_image_knuth(self):
        gi = GenderImage(face_detector = OcvCnnFacedetector(paddpercent=0.))
        df = gi('./media/20091020222328!KnuthAtOpenContentAlliance.jpg')
        self.assertEqual(len(df), 1)
        e = next(df.itertuples(index=False, name = 'useless'))
        self.assertEqual(e.bbox, (78, 46, 321, 289))
        self.assertEqual(e.sex_label, 'm')
        self.assertAlmostEqual(e.sex_decfunc, 6.615791, places=5)
        self.assertAlmostEqual(e.face_detect_conf, 0.99995565, places=4)

    def test_multifaceimage(self):
        gi = GenderImage()
        ret = gi('./media/800px-India_(236650352).jpg')
        self.assertEqual(len(ret), 5)
        # TODO: complete test later
