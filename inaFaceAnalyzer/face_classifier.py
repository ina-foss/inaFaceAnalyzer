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
import pandas as pd
import numbers
from abc import ABC, abstractmethod
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from .svm_utils import svm_load
from .opencv_utils import imread_rgb
from .remote_utils import get_remote


class FaceClassifier(ABC):
    """
    Abstract class to be implemented by face classifiers
    """
    @classmethod
    @abstractmethod
    def input_shape():
        """
        input image dimensions required by the classifier (width, height, depth)
        """
        pass

    @abstractmethod
    def list2batch(self, limg): pass

    @abstractmethod
    def inference(self, bfeats): pass

    @abstractmethod
    def decisionfunction2labels(self, df): pass

    @property
    def output_cols(self):
        if not hasattr(self, '_output_cols'):
            fake_input = [np.zeros(self.input_shape)]
            self._output_cols = list(self(fake_input, False).columns)
        return self._output_cols

    def average_results(self, df):
        if len(df) == 0:
            for c in self.output_cols:
                df[c] = []

        cols = [e for e in df.columns if e.endswith('_decfunc')]
        gbm = df.groupby('face_id')[cols].mean()
        gbm = self.decisionfunction2labels(gbm)

        return df.join(gbm, on='face_id', rsuffix='_avg')

    # Keras trick for async READ ?
    # bench execution time : time spent in read/exce . CPU vs GPU
    def imgpaths_batch(self, lfiles, return_features, batch_len=32):
        """
        images are assumed to be faces already detected, scaled, aligned, croped
        """
        assert len(lfiles) > 0

        if return_features:
            raise NotImplementedError()

        lbatchret = []

        for i in range(0, len(lfiles), batch_len):
            xbatch = [imread_rgb(e) for e in lfiles[i:(i+batch_len)]]

            lbatchret.append(self(xbatch, False)) # to change when return features will be managed

        return pd.concat(lbatchret).reset_index(drop=True)

    def __call__(self, limg, return_features):
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
        batch_ret_feats, batch_ret_preds = self.inference(self.list2batch(limg))
        batch_ret_preds = self.decisionfunction2labels(batch_ret_preds)

        if islist:
            if return_features:
                return batch_ret_feats, batch_ret_preds
            return batch_ret_preds

        ret = next(batch_ret_preds.itertuples(index=False, name='FaceClassifierResult'))
        if return_features:
            return  batch_ret_feats[0, :], ret
        return ret

class Resnet50FairFace(FaceClassifier):
    input_shape = (224, 224, 3)

    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-3].output])

    def list2batch(self, limg):
        x = np.concatenate([np.expand_dims(img_to_array(e), axis=0) for e in limg])
        return tensorflow.keras.applications.resnet50.preprocess_input(x)

    def inference(self, x):
        decisions, feats = self.model.predict(x)
        df = pd.DataFrame(decisions.ravel(), columns=['sex_decfunc'])
        return feats, df

    def decisionfunction2labels(self, df):
        df['sex_label'] = df.sex_decfunc.map(lambda x: 'm' if x > 0 else 'f' )
        return df


def _fairface_agedec2age(age_dec):
    ages = np.array([(0,2), (3,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69), (70, 79), (80,99)], dtype=np.float32)
    ages_mean = (np.sum(ages, axis=1) + 1) / 2.
    ages_range = ages[:, 1] - ages[:, 0] +1

    if isinstance(age_dec, numbers.Number):
        age_dec = np.array([age_dec])

    age_dec = np.array(age_dec)
    age_dec[age_dec <= -.5] = -.5 + 10**-8
    age_dec[age_dec >= 9.5] = 9.5 - 10**-8
    idec = np.round(age_dec).astype(np.int32)

    age_label = ages_mean[idec] + (age_dec - idec) * ages_range[idec]
    return age_label




class Resnet50FairFaceGRA(Resnet50FairFace):
    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface_GRA.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs + [m.layers[-5].output])

    def inference(self, x):
        gender, _, age, feats = self.model.predict(x)
        df = pd.DataFrame(zip(gender.ravel(), age.ravel()), columns=['sex_decfunc', 'age_decfunc'])
        return feats, df

    def decisionfunction2labels(self, df):
        df = super().decisionfunction2labels(df)
        df['age_label'] = _fairface_agedec2age(df.age_decfunc)
        return df

class OxfordVggFace(FaceClassifier):
    input_shape = (224, 224, 3)
    def __init__(self, hdf5_svm=None):
        # Face feature extractor from aligned and detected faces
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = self.input_shape, pooling ='avg')
        # SVM trained on VGG neural features
        if hdf5_svm is not None:
            self.gender_svm = svm_load(hdf5_svm)

    def decisionfunction2labels(self, df):
        df['sex_label'] = [self.gender_svm.classes_[1 if x > 0 else 0] for x in df.sex_decfunc]
        return df

    def list2batch(self, limg):
        """
        returns VGG16 Features
        limg is a list of preprocessed images supposed to be aligned and cropped and resized to 224*224
        """
        limg = [np.expand_dims(img_to_array(e[:, :, ::-1]), axis=0) for e in limg]
        x = utils.preprocess_input(np.concatenate(limg), version=1)
        return self.vgg_feature_extractor(x)

    def inference(self, x):
        return np.array(x), pd.DataFrame(self.gender_svm.decision_function(x), columns=['sex_decfunc'])

class Vggface_LSVM_YTF(OxfordVggFace):
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_ytf_zrezgui.hdf5'))


class Vggface_LSVM_FairFace(OxfordVggFace):
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_vgg16_fairface.hdf5'))
