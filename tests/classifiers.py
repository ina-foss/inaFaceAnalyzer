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
from inaFaceAnalyzer.face_classifier import Resnet50FairFace, Resnet50FairFaceGRA, Vggface_LSVM_YTF
from inaFaceAnalyzer.opencv_utils import imread_rgb


class TestClassifiers(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_single_image_single_output_vgg16(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Vggface_LSVM_YTF()
        ret = c(mat, True)
        self.assertEqual(len(ret), 2)
        feats, rett = ret

        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(rett.sex_label, str)
        self.assertIsInstance(rett.sex_decfunc, float)

    def test_2images_single_output_vgg16(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Vggface_LSVM_YTF()
        ret = c([mat, mat], True)
        self.assertEqual(len(ret), 2)
        feats, retdf = ret

        self.assertIsInstance(feats, np.ndarray)
        [self.assertIsInstance(e, str) for e in retdf.sex_label]
        [self.assertIsInstance(e, float) for e in retdf.sex_decfunc]


    def test_single_image_single_output_res50(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFace()
        feats ,ret = c(mat, True)
        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(ret.sex_label, str)
        self.assertIsInstance(ret.sex_decfunc, float)

    def test_single_image_multi_output(self):
        mat = np.zeros((224,224,3), dtype=np.uint8)
        c = Resnet50FairFaceGRA()
        feats, ret = c(mat, True)
        self.assertIsInstance(feats, np.ndarray)
        self.assertIsInstance(ret.sex_label, str)
        self.assertIsInstance(ret.age_label, float, type(ret.age_label))
        self.assertIsInstance(ret.age_decfunc, float, type(ret.age_label))
        self.assertIsInstance(ret.sex_decfunc, float, type(ret.age_label))


    def test_fairface_age_mapping(self):
        from inaFaceAnalyzer.face_classifier import _fairface_agedec2age
        # [(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70+)]
        # simple
        x = [0, 1, 2, 3, 9]
        y = np.array([1.5, 6.5, 15, 25, 90])
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x), decimal=5)
        # test limits
        x = [-1, -0.5, 9.5, 11]
        y = [0, 0, 100, 100]
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x), decimal=4)
        # harder - stuffs around centers...
        x = [-1/3./2, 4.5]
        y = [1, 40]
        np.testing.assert_almost_equal(y, _fairface_agedec2age(x), decimal=5)

    def test_preprocessed_img_list_singleoutput(self):
        c = Vggface_LSVM_YTF()
        df = c.preprocessed_img_list(['./media/diallo224.jpg', './media/knuth224.jpg', './media/diallo224.jpg'], False, batch_len=2)
        self.assertSequenceEqual(['f', 'm', 'f'], list(df.sex_label))
        np.testing.assert_almost_equal([-3.1886155, 6.7310688, -3.1886155], df.sex_decfunc, decimal=5)

    def test_preprocessed_img_list_multioutput(self):
        c = Resnet50FairFaceGRA()
        df = c.preprocessed_img_list(['./media/diallo224.jpg', './media/knuth224.jpg', './media/diallo224.jpg'], False, batch_len=2)
        ref_genderL = ['f', 'm', 'f']
        ref_ageL =[25.723361, 61.890726, 25.723361]
        ref_genderD = [-5.632368, 7.2553654, -5.632368]
        ref_ageD = [3.0723362, 6.6890726, 3.0723362]
        self.assertSequenceEqual(ref_genderL, list(df.sex_label))
        np.testing.assert_almost_equal(ref_ageL, df.age_label, decimal=5)
        np.testing.assert_almost_equal(ref_genderD, df.sex_decfunc, decimal=5)
        np.testing.assert_almost_equal(ref_ageD, df.age_decfunc, decimal=5)

    def test_batch_order(self):
        # test if image position in batch has an impact on decision value
        limg = [imread_rgb('./media/diallo224.jpg'), imread_rgb('./media/knuth224.jpg')] * 32
        c = Resnet50FairFace()
        retdf = c(limg, False)
        d1 = retdf.sex_decfunc[::2].reset_index(drop=True)
        d2 = retdf.sex_decfunc[1::2].reset_index(drop=True)
        np.testing.assert_almost_equal([d1[0]] * 32, d1, decimal=3)
        np.testing.assert_almost_equal([d2[0]] * 32, d2, decimal=3)

    def test_racelayerdeleted(self):
        # test if "race" prediction layer is set to NaN in the public distribution
        c = Resnet50FairFaceGRA()
        racelayer = c.model.layers[-2]
        w, b = racelayer.weights
        assert np.all(w.numpy() != w.numpy())
        assert np.all(b.numpy() != b.numpy())
