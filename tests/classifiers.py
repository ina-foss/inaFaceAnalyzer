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
import tensorflow as tf
import numpy as np
from inaFaceGender.face_classifier import Resnet50FairFace, Resnet50FairFaceGRA, Vggface_LSVM_YTF


class TestClassifiers(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_single_image_single_output_vgg16(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Vggface_LSVM_YTF()
        ret = c(mat)
        self.assertEqual(len(ret), 3)
        feats, label, dec = ret
        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(label, str)
        self.assertIsInstance(dec, float)


    def test_single_image_single_output_res50(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFace()
        ret = c(mat)
        self.assertEqual(len(ret), 3)
        feats, label, dec = ret
        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(label, str)
        self.assertIsInstance(dec, np.float32)

    def test_single_image_multi_output(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFaceGRA()
        ret = c(mat)
        self.assertEqual(len(ret), 5)
        feats, genderL, ageL, genderD, ageD = ret
        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(genderL, str)
        self.assertIsInstance(ageL, np.float32, type(ageL))
        self.assertIsInstance(ageD, np.float32, type(ageD))
        self.assertIsInstance(genderD, np.float32, type(genderD))


    def test_fairface_age_mapping(self):
        from inaFaceGender.face_classifier import _fairface_agedec2age
        # [(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70+)]
        # simple
        x = [0, 1, 2, 3, 9]
        y = np.array([1.5, 6.5, 15, 25, 90])
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x))
        # test limits
        x = [-1, -0.5, 9.5, 11]
        y = [0, 0, 100, 100]
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x))
        # harder - stuffs around centers...
        x = [-1/3./2, 4.5]
        y = [1, 40]
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x))

    def test_imgpaths_batch_singleoutput(self):
        c = Vggface_LSVM_YTF()
        f, l, d = c.imgpaths_batch(['./media/diallo224.jpg', './media/knuth224.jpg', './media/diallo224.jpg'], batch_len=2)
        self.assertSequenceEqual(['f', 'm', 'f'], l)
        np.testing.assert_almost_equal([-3.1886155, 6.7310688, -3.1886155], d, decimal=5)

    def test_imgpaths_batch_multioutput(self):
        c = Resnet50FairFaceGRA()
        f, gl, al, gd, ad = c.imgpaths_batch(['./media/diallo224.jpg', './media/knuth224.jpg', './media/diallo224.jpg'], batch_len=2)
        ref_genderL = ['f', 'm', 'f']
        ref_ageL =[25.72336197, 61.89072609, 25.72336197]
        ref_genderD = [-5.632368, 7.2553654, -5.632368]
        ref_ageD = [3.0723362, 6.6890726, 3.0723362]
        self.assertSequenceEqual(ref_genderL, gl)
        np.testing.assert_almost_equal(ref_ageL, al, decimal=5)
        np.testing.assert_almost_equal(ref_genderD, gd, decimal=5)
        np.testing.assert_almost_equal(ref_ageD, ad, decimal=5)
