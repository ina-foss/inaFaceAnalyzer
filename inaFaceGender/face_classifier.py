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

import numpy as np
import numbers
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from .svm_utils import svm_load
from .opencv_utils import imread_rgb
from .remote_utils import get_remote


# TODO : return dictionary with label & decision function name
# this may require additional refactoring

class AbstractFaceClassifier:

    ## SOON/ALREADY DEPRECATED => NO: put in tests
    # Keras trick for async READ ?
    # return dict or result object ?

    # bench execution time : time spent in read/exce . CPU vs GPU
    # use a Keras sequence feature extraction method ??
    def imgpaths_batch(self, lfiles, batch_len=32):
        """
        images are assumed to be faces already detected, scaled, aligned, croped
        """
        lret = []

        for i in range(0, len(lfiles), batch_len):
            xbatch = [imread_rgb(e) for e in lfiles[i:(i+batch_len)]]
            lret.append(self(xbatch))

        feats = np.concatenate([e[0] for e in lret])

        # FIXME HERE WITH multiple outputs!
        #print(lret)
        labels = [lab for e in lret for lab in e[1]]
        decisions = [dec for e in lret for dec in e[2]]

        return feats, labels, decisions

    def __call__(self, limg):
        """
        Classify a list of images
        images are supposed to be preprocessed faces: aligned, cropped
        Parameters
        ----------
        limg : list of images, a single image can also be used

        Returns
        -------
        feats :
            face features used as input to the final classifier
        label : str
            f for female, m for male
        decision_value : float
            decision function value (negative for female, positive for male)
        """

        if isinstance(limg, list):
            islist = True
        else:
            islist = False
            limg = [limg]

        assert np.all([e.shape == self.input_shape for e in limg])
        feats, labels, decisions = self.inference(self.list2batch(limg))

        if islist:
            return feats, labels, decisions

        #print('call', labels, labels[0])
        if isinstance(labels, tuple):
            labels = tuple([e[0] for e in labels])
        else:
            labels = labels[0]
        if isinstance(decisions, tuple):
            decisions = tuple([e[0] for e in decisions])
        else:
            decisions = decisions[0]
        return feats[0], labels, decisions

    def list2batch(self, limg):
        raise NotImplementedError()

    def inference(self, bfeats):
       raise NotImplementedError()


class Resnet50FairFace(AbstractFaceClassifier):
    input_shape = (224, 224, 3)
    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-3].output])

    def list2batch(self, limg):
        x = np.concatenate([np.expand_dims(img_to_array(e), axis=0) for e in limg])
        return tensorflow.keras.applications.resnet50.preprocess_input(x)

    def inference(self, x):
        decisions, feats = self.model.predict(x)
        decisions = decisions.ravel()
        assert len(decisions) == len(x)
        labels = ['m' if e > 0 else 'f' for e in decisions]
        return feats, labels, decisions


def _fairface_agedec2age(age_dec):
    ages = np.array([(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70, 79), (80,99)])
    ages_mean = (np.sum(ages, axis=1) + 1) / 2.
    ages_range = ages[:, 1] - ages[:, 0] +1

    if isinstance(age_dec, numbers.Number):
        age_dec = np.array([age_dec], dtype=np.float32)

    age_dec = np.array(age_dec)
    age_dec[age_dec <= -.5] = -.499999999
    age_dec[age_dec >= 9.5] = 9.499999999
    idec = np.round(age_dec).astype(np.int32)

    age_label = ages_mean[idec] + (age_dec - idec) * ages_range[idec]
    return age_label

class Resnet50FairFaceGRA(Resnet50FairFace):

    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface_GRA.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-5].output])

    def inference(self, x):
        gender, _, age, feats = self.model.predict(x)

        # TODO gender stuffs - similar to mother class => factorize
        gender_dec = gender.ravel()
        assert(len(gender_dec) == len(x))
        gender_labels = ['m' if e > 0 else 'f' for e in gender_dec]

        age_dec = age.ravel()
        age_labels = _fairface_agedec2age(age_dec)

        return feats, (gender_labels, age_labels), (gender_dec, age_dec)


class OxfordVggFace(AbstractFaceClassifier):
    input_shape = (224, 224, 3)
    def __init__(self, hdf5_svm=None):
        # Face feature extractor from aligned and detected faces
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = self.input_shape, pooling ='avg')
        # SVM trained on VGG neural features
        if hdf5_svm is not None:
            self.gender_svm = svm_load(hdf5_svm)

    def list2batch(self, limg):
        """

        returns VGG16 Features
        limg is a list of preprocessed images supposed to be aligned and cropped and resized to 224*224
        """
        limg = [np.expand_dims(img_to_array(e[:, :, ::-1]), axis=0) for e in limg]
        x = utils.preprocess_input(np.concatenate(limg), version=1)
        return self.vgg_feature_extractor(x)

    def inference(self, x):
        decisions = self.gender_svm.decision_function(x)
        labels = [self.gender_svm.classes_[1 if x > 0 else 0] for x in decisions]
        return np.array(x), labels, decisions


class Vggface_LSVM_YTF(OxfordVggFace):
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_ytf_zrezgui.hdf5'))


class Vggface_LSVM_FairFace(OxfordVggFace):
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_vgg16_fairface.hdf5'))
