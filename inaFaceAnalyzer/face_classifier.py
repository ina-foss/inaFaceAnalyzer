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

"""
Module :mod:`inaFaceAnalyzer.face_classifier` define classes providing
pretrained DNN face classification models allowing to predict gender and/or age from faces.

Four classes are currently proposed :

- :class:`Resnet50FairFaceGRA` predicts age and gender from faces, and is \
    associated to the best classification performances. It should be used by default.
- :class:`Resnet50FairFace`, :class:`Vggface_LSVM_YTF` and \
    :class:`Vggface_LSVM_FairFace` are provided for reproducibility reasons \
    and predict gender only.

Face classification classes share a common interface defined in abstracty class :class:`FaceClassifier`.
They can be used with methods  :

- :meth:`FaceClassifier.preprocessed_img_list` for processing image lists stored on disk
- :meth:`FaceClassifier.__call__` for processing list of image frames.

.. warning :: Face classifiers assume input images contain a single detected,
    centered, eye-aligned, scaled and preprocessed face of dimensions 224*224 pixels


>>> from inaFaceAnalyzer.face_classifier import Resnet50FairFaceGRA
>>> classif = Resnet50FairFaceGRA()
>>> classif.preprocessed_img_list(['./media/diallo224.jpg', './media/knuth224.jpg'])
                filename  sex_decfunc  age_decfunc sex_label  age_label
0  ./media/diallo224.jpg    -5.632371     3.072337         f  25.723367
1   ./media/knuth224.jpg     7.255364     6.689072         m  61.890717

"""

import numpy as np
import pandas as pd
import numbers
from abc import ABC, abstractmethod
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

import inaFaceAnalyzer.keras_vggface_patch as keras_vggface
from .svm_utils import svm_load
from .opencv_utils import imread_rgb, disp_frame
from .remote_utils import get_remote


class FaceClassifier(ABC):
    """
    Abstract class to be implemented by face classifiers
    """

    # The 3 properties bellow (input_shape, bbox_scale, bbox2square) are
    # currently common to all implemented face classifiers
    # they provide information on the face preprocessing steps used for
    # training the classification models
    # in future, they may be defined separately for each classifier using
    # abstract properties

    #: input image dimensions required by the classifier (height, width, depth)
    input_shape = (224, 224, 3)

    #: implemented classifiers are optimized for a given scale factor to be
    #: applied on face bounding boxes to be defined here
    bbox_scale = 1.1

    #: implemented face classifiers may require a preprocessing step consisting
    #: to extend the face bounding box such as the resulting box is the smallest
    #: square containing the detected face
    bbox2square = True


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

    # TODO : Keras trick for async READ ?
    # bench execution time : time spent in read/exce . CPU vs GPU
    # TODO: add progress bar with verbose option
    def preprocessed_img_list(self, lfiles, batch_len=32):
        """
        Performs classification on a list of preprocessed face images
        Preprocessed face images are assumed to contain a single face which is
        already detected, cropped, aligned and scaled to classifier's input
        dimensions (for now: 224*224 pixels)

        Args:
            lfiles (list): list of image paths: ['/path/to/img1', '/path/to/img2']
            batch_len (int, optional): DNN batch size. Larger batch_len results
                in faster processing times.
                Batch lenght is dependent on available GPU memory.
                Defaults to 32 (suitable for a laptop GPU).

        Returns:
            pandas.DataFrame. a DataFrame with one record for each input image
        """
        assert len(lfiles) > 0

        lbatchret = []

        for i in range(0, len(lfiles), batch_len):
            xbatch = [imread_rgb(e) for e in lfiles[i:(i+batch_len)]]

            lbatchret.append(self(xbatch, False)) # to change when return features will be managed

        df = pd.concat(lbatchret).reset_index(drop=True)
        df.insert(0, 'filename', lfiles)
        return df

    def __call__(self, limg, verbose=False):        
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

        # """
        # Classify a list of images
        # images are supposed to be preprocessed faces: aligned, cropped
        
        # Args:
        #     limg (list): list of images (224*224*3), a single image can also be used 
        #     verbose (TYPE, optional): display intermediate prediction results. Defaults to False.

        # Returns:
        #     :class:`pandas.DataFrame` : a dataframe with one row per image.
        #         column names are of the form <information>_<output_type> with
        #         - <information> in {gender, age} corresponding to the information being predicted
        #         - <output_type> in {decfunc,label}  with decfunc

        # """
        
        


        if isinstance(limg, list):
            islist = True
        else:
            islist = False
            limg = [limg]

        assert np.all([e.shape == self.input_shape for e in limg])
        batch_ret_preds = self.inference(self.list2batch(limg))
        batch_ret_preds = self.decisionfunction2labels(batch_ret_preds)

        if verbose:
            for img, pred in zip(limg, batch_ret_preds.itertuples(index=False, name='FaceClassifierResult')):
                disp_frame(img)
                print('prediction', pred)

        if islist:
            return batch_ret_preds

        ret = next(batch_ret_preds.itertuples(index=False, name='FaceClassifierResult'))
        return ret

class Resnet50FairFace(FaceClassifier):
    """
    Resnet50FairFace uses Resnet50 architecture trained to predict gender on
    `FairFace <https://github.com/joojs/fairface>`_.
    """
    
    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs)

    def list2batch(self, limg):
        x = np.concatenate([np.expand_dims(img_to_array(e), axis=0) for e in limg])
        return tensorflow.keras.applications.resnet50.preprocess_input(x)

    def inference(self, x):
        decisions = self.model.predict(x)
        df = pd.DataFrame(decisions.ravel(), columns=['sex_decfunc'])
        return df

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
    minval = -.5 + 10**-6
    age_dec[age_dec < minval] = minval
    maxval = 9.5 - 10**-6
    age_dec[age_dec > maxval] = maxval
    idec = np.round(age_dec).astype(np.int32)

    age_label = ages_mean[idec] + (age_dec - idec) * ages_range[idec]
    return age_label

class Resnet50FairFaceGRA(Resnet50FairFace):
    """
    Resnet50FairFaceGRA predicts age and gender and is the most accurate proposed.
    It uses Resnet50 architecture and is trained to predict gender, age and race on
    `FairFace <https://github.com/joojs/fairface>`_.
    After consultation of French CNIL (French data protection authority) and
    DDD (French Rights Defender), racial classification layers were erased
    from this public distribution in order to prevent their use for non ethical purposes.
    These models can however be provided for free after examination of each demand.
    """

    def __init__(self):
        m = keras.models.load_model(get_remote('keras_resnet50_fairface_GRA.h5'), compile=False)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs)

    def inference(self, x):
        gender, _, age = self.model.predict(x)
        df = pd.DataFrame(zip(gender.ravel(), age.ravel()), columns=['sex_decfunc', 'age_decfunc'])
        return df

    def decisionfunction2labels(self, df):
        df = super().decisionfunction2labels(df)
        df['age_label'] = _fairface_agedec2age(df.age_decfunc)
        return df

# Fair Face Dataset class labels
# Classification model can be provided upon request
_race_cols = ['black_decfunc',
              'eastasian_decfunc',
              'indian_decfunc',
              'latinohispanic_decfunc',
              'middleeastern_decfunc',
              'southeastasian_decfunc',
              'white_decfunc']

class Resnet50FairFaceGRAFull(Resnet50FairFaceGRA):
    """
    Resnet50FairFaceGRAFull predicts age, gender and perceived origin (race).
    It uses Resnet50 architecture trained on `FairFace <https://github.com/joojs/fairface>`_.
    After consultation of French CNIL (French data protection authority) and
    DDD (French Rights Defender), model weights cannot be made publicly available
    in order to prevent its use for non ethical purposes. They can be provided
    for free after examination of each demand.
    The current class will throw an exception if used without the models.
    """
    def __init__(self):
        try:
            m = keras.models.load_model(get_remote('keras_resnet50_fairface_GRA-full.h5'), compile=False)
        except:
            msg = """Racial classification models are not publicly available.
            Please contact maintainers or use another classification model.
            """
            raise Exception(msg)
        self.model = tensorflow.keras.Model(inputs=m.inputs, outputs=m.outputs)

    def inference(self, x):
        gender, race, age = self.model.predict(x)
        tmp = np.concatenate([gender, age, race], axis=1)
        return pd.DataFrame(tmp, columns= ['sex_decfunc', 'age_decfunc'] + _race_cols)

    def decisionfunction2labels(self, df):
        df = super().decisionfunction2labels(df)
        racedf = df[_race_cols]
        df['race_label'] = racedf.idxmax(axis='columns').map(lambda x: x.split('_')[0])
        return df


class OxfordVggFace(FaceClassifier):
    '''
    OxfordVggFace instances are based on VGG16 architectures
    pretrained using a triplet loss paradigm allowing to obtain face neural
    representation, that we use to train linear SVM classification systems.

    This class takes advantage of Refik Can Malli's keras-vggface module,
    providing pretrained VGG16 models
    https://github.com/rcmalli/keras-vggface
    '''

    def __init__(self, hdf5_svm=None):
        """
        Constructor 

        Args:
            hdf5_svm (str, optional): path to serialized SVM . Defaults to None.

        """
        # Face feature extractor from aligned and detected faces
        self.vgg_feature_extractor = keras_vggface.VGG16(self.input_shape)
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
        x = keras_vggface.preprocess_input(np.concatenate(limg))
        return self.vgg_feature_extractor(x)

    def inference(self, x):
        return pd.DataFrame(self.gender_svm.decision_function(x), columns=['sex_decfunc'])

class Vggface_LSVM_YTF(OxfordVggFace):
    """
    Vggface_LSVM_FairFace predict gender from face using pretrained Oxford VGG16
    facial embedings used to train a Linear SVM on `Youtube Faces DB <https://www.cs.tau.ac.il/~wolf/ytfaces/>`_.

    This method is fully described in Zohra Rezgui's internship report at INA:
    Détection et classification de visages pour la description de l’égalité
    femme-homme dans les archives télévisuelles, Higher School of Statistics
    and Information Analysis, University of Carthage, 2019    
    """
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_ytf_zrezgui.hdf5'))


class Vggface_LSVM_FairFace(OxfordVggFace):
    """
    Vggface_LSVM_FairFace predict gender from face using pretrained Oxford VGG16
    facial embedings used to train a Linear SVM on `FairFace <https://github.com/joojs/fairface>`_.
    """
    def __init__(self):
        OxfordVggFace.__init__(self, get_remote('svm_vgg16_fairface.hdf5'))

help = '''face classifier to be used in the analysis:
Resnet50FairFaceGRA predicts age and gender and is the most accurate.
It uses Resnet50 architecture and is trained to predict gender, age and race on FairFace.
After consultation of French CNIL (French data protection authority) and
DDD (French Rights Defender), racial classification layers were erased
from this public distribution in order to prevent their use for non ethical purposes.
These models can however be provided for free after examination of each demand.
Resnet50FairFace only predicts gender, and is trained on FairFace with a Resnet50 architecture.
Vggface_LSVM_YTF predicts only gender. It uses an Oxford VGG 16 neural representation
of faces combined with a linear SVM that was trained on Youtube Faces database
by Zohra Rezgui during her internship at INA. It was used in digital earlier humanities studies.
Vggface_LSVM_Fairface has the same architecture and equivalent performances than
Vggface_LSVM_YTF. Its linear SVM model was trained on FairFace.
'''
choices = ['Resnet50FairFaceGRA', 'Vggface_LSVM_YTF', 'Resnet50FairFace', 'Vggface_LSVM_FairFace']
def faceclassifier_cmdline(parser):
    parser.add_argument ('--classifier', default='Resnet50FairFaceGRA',
                         choices = choices, help = help)

def faceclassifier_factory(args):
    if args.classifier == 'Resnet50FairFaceGRA':
        return Resnet50FairFaceGRA()
    if args.classifier == 'Resnet50FairFace':
        return Resnet50FairFace()
    if args.classifier == 'Vggface_LSVM_FairFace':
        return Vggface_LSVM_FairFace()
    if args.classifier == 'Vggface_LSVM_YTF':
        return Vggface_LSVM_YTF()
