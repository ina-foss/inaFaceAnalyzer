#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2021 Ina (David Doukhan & Zohra Rezgui- http://www.ina.fr/)

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

import os
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
import tensorflow
from tensorflow import keras
from .svm_utils import svm_load

class Resnet50FairFace:
    input_shape = (224, 224)
    def __init__(self):
        p = os.path.dirname(os.path.realpath(__file__))
        self.model = keras.models.load_model(p + '/models/keras_resnet50_fairface.h5', compile=False)
    def __call__(self, img):
        x = tensorflow.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tensorflow.keras.applications.resnet50.preprocess_input(x)
        ret = self.model.predict(x)
        label = 'm' if ret > 0 else 'f'
        return None, label, ret

class VGG16_LinSVM:
    input_shape = (224,224)
    def __init__(self):
        p = os.path.dirname(os.path.realpath(__file__)) + '/models/'

        # Face feature extractor from aligned and detected faces
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = (224, 224, 3), pooling ='avg')

        # SVM trained on neural features - dependent on the neural representation
        # should be packed together
        self.gender_svm = svm_load(p + 'svm_classifier.hdf5')
        # f = h5py.File(p + 'svm_classifier.hdf5', 'r')
        # svm = LinearSVC()
        # svm.classes_ = np.array(f['linearsvc/classes'][:]).astype('<U1')
        # svm.intercept_ = f['linearsvc/intercept'][:]
        # svm.coef_ = f['linearsvc/coef'][:]
        # self.gender_svm = svm

    def extract_features(self, img):
        """
        returns VGG16 Features
        img is supposed to be aligned and cropped and resized to 224*224
        """
        assert (img.shape[0], img.shape[1]) == (224, 224)
        img  =  img[:, :, ::-1] # RGB to something else ??
        img = image.img_to_array(img)
        img = utils.preprocess_input(img, version=1)
        img = np.expand_dims(img, axis=0)
        return self.vgg_feature_extractor.predict(img)

    def __call__(self, img):
        """
        Parameters
        ----------
        #TODO: find class name
        img : opencv frame
            img is supposed to be aligned and cropped and resized to 224*224

        Returns
        -------
        feats :
            face features used as input to the final classifier
        label : str
            f for female, m for male
        decision_value : float
            decision function value (negative for female, positive for male)

        """
        feats = self.extract_features(img)
        #label = self.gender_svm.predict(feats)[0]
        decision_value = self.gender_svm.decision_function(feats)[0]
#        print(decision_value)
        label = self.gender_svm.classes_[1 if decision_value > 0 else 0]
        return feats, label, decision_value
