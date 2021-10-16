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
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import get_file
from .svm_utils import svm_load
from .opencv_utils import imread_rgb


# TODO: batch method should be used by default
# __call__ should be removed in a nearby future


class AbstractFaceClassifier:
    models_url = 'https://github.com/ina-foss/inaFaceGender/releases/download/models-init/'
    def imgpaths_batch(self, lfiles, batch_len=32):
        """
        images are assumed to be faces already detected, scaled, aligned, croped
        """
        lret = []
        while lfiles:
#            print('batch')
            limg = [imread_rgb(e) for e in lfiles[:batch_len]]
            lfiles = lfiles[batch_len:]
            lret.append(self.batch(limg))
        feats = np.concatenate([e[0] for e in lret])
        labels = [lab for e in lret for lab in e[1]]
        decisions = [dec for e in lret for dec in e[2]]
        return feats, labels, decisions


class Resnet50FairFaceGRA(AbstractFaceClassifier):
    input_shape = (224, 224)
    def __init__(self):
        url = self.models_url + 'keras_resnet50_fairface_GRA.h5'
        fname = get_file('keras_resnet50_fairface_GRA.h5', url)
        m = keras.models.load_model(fname, compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-5].output])

    def __call__(self, img):
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tensorflow.keras.applications.resnet50.preprocess_input(x)
        print( self.model.predict(x))
        gender, _, age, feats = self.model.predict(x)
        gender_dec = gender.ravel()
        assert(len(gender_dec) == 1)
        gender_dec = gender_dec[0]
        gender_label = 'm' if gender_dec > 0 else 'f'
        age_dec = age[0][0]

        lage = [(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70, 89), (90,99)]
        if age_dec < 0:
            age_label = 0.
        elif int(np.round(age_dec)) > 9:
            age_label = 100.
        else:
            start, stop = lage[int(np.round(age_dec))]
            mean = (start + stop + 1) / 2.
            age_label = mean + (age_dec - np.round(age_dec)) * (stop -start+1)
        return feats, (gender_label, age_label), (gender_dec, age_dec)


class Resnet50FairFace(AbstractFaceClassifier):
    input_shape = (224, 224)
    def __init__(self):
        url = self.models_url + 'keras_resnet50_fairface.h5'
        fname = get_file('keras_resnet50_fairface.h5', url)
        m = keras.models.load_model(fname, compile=False)        
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-3].output])
    def __call__(self, img):
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tensorflow.keras.applications.resnet50.preprocess_input(x)
        ret, feats = self.model.predict(x)
        ret = ret.ravel()
        assert len(ret) == 1
        ret = ret[0]
        label = 'm' if ret > 0 else 'f'
        return feats, label, ret
    def batch(self, limg):
        x = np.concatenate([np.expand_dims(img_to_array(e), axis=0) for e in limg])
        x = tensorflow.keras.applications.resnet50.preprocess_input(x)
        decisions, feats = self.model.predict(x)
        decisions = decisions.ravel()
        assert len(decisions) == len(limg)
        labels = ['m' if e > 0 else 'f' for e in decisions]
        #print(decisions.shape)
        return feats, labels, decisions

class VGG16_LinSVM(AbstractFaceClassifier):
    input_shape = (224,224)
    def __init__(self):
        p = os.path.dirname(os.path.realpath(__file__)) + '/models/'

        # Face feature extractor from aligned and detected faces
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = (224, 224, 3), pooling ='avg')

        # SVM trained on VG neural features
        self.gender_svm = svm_load(p + 'svm_classifier.hdf5')


    def extract_features(self, img):
        """
        returns VGG16 Features
        img is supposed to be aligned and cropped and resized to 224*224
        """
        assert (img.shape[0], img.shape[1]) == (224, 224)
        img  =  img[:, :, ::-1] # RGB to something else ??
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = utils.preprocess_input(img, version=1)
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
        decision_value = self.gender_svm.decision_function(feats)[0]
        label = self.gender_svm.classes_[1 if decision_value > 0 else 0]
        return feats, label, decision_value

    def batch(self, limg):
        ltmpimg = []
        for img in limg:
            assert (img.shape[0], img.shape[1]) == (224, 224)
            img  =  img[:, :, ::-1] # RGB to something else ??
            img = img_to_array(img)
            ltmpimg.append(np.expand_dims(img, axis=0))
        x = utils.preprocess_input(np.concatenate(ltmpimg), version=1)
        feats = self.vgg_feature_extractor(x)
        decision = self.gender_svm.decision_function(feats)
        labels = [self.gender_svm.classes_[1 if x > 0 else 0] for x in decision]
        return feats, labels, decision
