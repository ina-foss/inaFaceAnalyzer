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
from inaFaceAnalyzer.inaFaceAnalyzer import ImageAnalyzer
from inaFaceAnalyzer.face_detector import OcvCnnFacedetector
from inaFaceAnalyzer.face_classifier import Resnet50FairFaceGRA, Vggface_LSVM_YTF

class TestSingleImage(unittest.TestCase):

    def tearDown(self):
        tf.keras.backend.clear_session()

   # IMAGE

    def test_image_all_diallo(self):
        gi = ImageAnalyzer(face_detector = OcvCnnFacedetector(padd_prct=0.),
                         face_classifier = Vggface_LSVM_YTF())
        df = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(df), 1)
        self.assertEqual(df.bbox[0], (432, 246, 989, 803))
        self.assertEqual(df.sex_label[0], 'f')
        self.assertAlmostEqual(df.sex_decfunc[0], -1.16580753, places=5)
        self.assertAlmostEqual(df.detect_conf[0], 0.99964356, places=4)

    def test_image_all_diallo_multioutput(self):
        gi = ImageAnalyzer(face_classifier = Resnet50FairFaceGRA(),
                           face_detector = OcvCnnFacedetector())
        df = gi('./media/Europa21_-_2.jpg')
        self.assertEqual(len(df), 1)
        e = next(df.itertuples(index=False, name = 'useless'))
        self.assertEqual(e.bbox, (421, 235, 997, 811))
        self.assertEqual(e.sex_label, 'f')
        self.assertAlmostEqual(e.age_label, 22.44171142, places=2)
        self.assertAlmostEqual(e.sex_decfunc, -4.708023, places=2)
        self.assertAlmostEqual(e.age_decfunc, 2.744171142, places=2)
        self.assertAlmostEqual(e.detect_conf, 0.987, places=2)


    def test_image_knuth(self):
        gi = ImageAnalyzer(face_detector = OcvCnnFacedetector(padd_prct=0.),
                         face_classifier = Vggface_LSVM_YTF())
        df = gi('./media/20091020222328!KnuthAtOpenContentAlliance.jpg')
        self.assertEqual(len(df), 1)
        e = next(df.itertuples(index=False, name = 'useless'))
        self.assertEqual(e.bbox, (79, 47, 322, 290))
        self.assertEqual(e.sex_label, 'm')
        self.assertAlmostEqual(e.sex_decfunc, 8.734114471, places=5)
        self.assertAlmostEqual(e.detect_conf, 0.99995565, places=4)

    def test_multifaceimage(self):
        gi = ImageAnalyzer(face_detector = OcvCnnFacedetector())
        ret = gi('./media/800px-India_(236650352).jpg')
        self.assertEqual(len(ret), 5)
        # TODO: complete test later
