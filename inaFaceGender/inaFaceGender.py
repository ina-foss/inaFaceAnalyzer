#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019 Ina (Zohra Rezgui & David Doukhan - http://www.ina.fr/)

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
from .opencv_utils import video_iterator, imread_rgb
from .face_tracking import TrackerDetector
from .face_detector import OcvCnnFacedetector
from .face_classifier import Vggface_LSVM_YTF
from .face_alignment import Dlib68FaceAlignment
from .face_preprocessing import preprocess_face



class AbstractGender:
    """
    This is an abstract class containg the common code to be used to process
    images, videos, with/without tracking
    """
    batch_len = 32

    def __init__(self, face_detector, face_classifier, bbox_scaling, squarify_bbox, verbose):
        """
        Constructor
        Parameters
        ----------
        face_detector : instance of face_detector.OcvCnnFacedetector or None
            More face detections modules may be implemented
            if None, then manual bounding boxes should be provided
        bbox_scaling : float
            scaling factor to be applied to the face bounding box.
            larger bounding box may help for sex classification from face
        squarify_bbox : boolean
            if set to True, then the bounding box (manual or automatic) is set to a square
        verbose : boolean
            If True, will display several usefull intermediate images and results
        """
        # face detection system
        self.face_detector = face_detector

        # set all bounding box shapes to square
        self.squarify_bbox = squarify_bbox

        # scaling factor to be applied to face bounding boxes
        self.bbox_scaling = bbox_scaling

        # face alignment module
        self.face_alignment = Dlib68FaceAlignment()

        # Face feature extractor from aligned and detected faces
        self.classifier = face_classifier

        # True if some verbose is required
        self.verbose = verbose




    #TODO : test in multi output
    # may be deprecated in a near future since it does not takes advantage of batches
    def classif_from_frame_and_bbox(self, frame, bbox, bbox_square, bbox_scale, bbox_norm):

        oshape = self.classifier.input_shape[:-1]
        fa, vrb = (self.face_alignment, self.verbose)
        face_img, bbox = preprocess_face(frame, bbox, bbox_square, bbox_scale, bbox_norm, fa, oshape, vrb)

        ret = self.classifier(face_img)
        # dirty trick used for retro compatibility
        # should return ret + [bbox]
        return [ret[0], bbox] + ret[1:]

    def detect_and_classify_faces_from_frame(self, frame):
        ret = []
        for bb, detect_conf in self.face_detector(frame):
            if self.verbose:
                print('bbox: %s, conf: %f' % (bb, detect_conf))
            ret.append(self.classif_from_frame_and_bbox(frame, bb, self.squarify_bbox, self.bbox_scaling, True) + [detect_conf])

            if self.verbose:
                ## TO CHANGE
                print('bounding box (x1, y1, x2, y2),' + ','.join(self.classifier.outnames[1:]) + ',face detection confidence')
                print(ret[-1][1:])
                print()
        return ret


    # THIS METHOD SHOULD BE REMOVED IN A NEAR FUTRE
    # it is removed in __call__
    # it should be now removed in detect with tracking
    def process_batch(self, lbatch):
        # TODO: is it possible to provide only list of images ???
        # lbatch cpontains tuples (iframe, bb (original bounding box), detect_conf, face_img, bbox)

        # TODO : Batch should be already set to its size of max batchlen
        batch = lbatch[:self.batch_len]


        classif_ret = self.classifier([e[3] for e in batch])
        classif_desc = ','.join(self.classifier.outnames[1:])

        # TO BE DONE OUTSIDE
        info = []
        for i, (iframe, _, detect_conf, _, bbox) in enumerate(batch):
            info.append((iframe, bbox, *[e[i] for e in classif_ret[1:]], detect_conf))
            if self.verbose:
                print('iframe, bounding box (x1, y1, x2, y2), %s, face detection confidence' % classif_desc)
                last = info[-1]
                print(*last)
                print()

        # TODO : should not return the updated batch....
        return lbatch[self.batch_len:], info



class GenderImage(AbstractGender):
    def __init__(self, face_detector = OcvCnnFacedetector(), face_classifier=Vggface_LSVM_YTF(), bbox_scaling=1.1, squarify=True, verbose = False):
        AbstractGender.__init__(self, face_detector, face_classifier, bbox_scaling, squarify, verbose)



    def __call__(self, img_path):
        frame = imread_rgb(img_path, self.verbose)
        return self.detect_and_classify_faces_from_frame(frame)


class GenderVideo(AbstractGender):
    """
    This is a class regrouping all phases of a pipeline designed for gender classification from video.

    Attributes:
        face_detector: Face detection model.
        face_alignment: Face alignment model.
        gender_svm: Gender SVM classifier model.
        vgg_feature_extractor: VGGFace neural model used for feature extraction.
        threshold: quality of face detection considered acceptable, value between 0 and 1.
    """
    def __init__(self, face_detector = OcvCnnFacedetector(), face_classifier=Vggface_LSVM_YTF(), bbox_scaling=1.1, squarify=True, verbose = False):
        AbstractGender.__init__(self, face_detector, face_classifier, bbox_scaling, squarify, verbose)


    def __call__(self, video_path, subsamp_coeff = 1 ,  offset = -1):

        """
        Pipeline function for gender classification from videos without tracking.

        Parameters:
            video_path (string): Path for input video.
            subsamp_coeff (int) : only 1/subsamp_coeff frames will be processed
            offset (float) : Time in milliseconds to skip at the beginning of the video.


        Returns:
            info: A Dataframe with frame and face information (coordinates, decision function,labels..)
        """


        oshape = self.classifier.input_shape[:-1]
        #classif_desc = ','.join(self.classifier.outnames[1:])

        info = []
        lbatch_img = []
        lbatch_info = []

        for iframe, frame in video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=min(offset, 0), verbose=self.verbose):

            for bb, detect_conf in self.face_detector(frame):
                if self.verbose:
                    print('bbox: %s, conf: %f' % (bb, detect_conf))


                face_img, bbox = preprocess_face(frame, bb, self.squarify_bbox, self.bbox_scaling, True, self.face_alignment, oshape, self.verbose)

                #lbatch.append((iframe, bb, detect_conf, face_img, bbox))
                lbatch_info.append([iframe, bbox, detect_conf])
                lbatch_img.append(face_img)

            while len(lbatch_img) > self.batch_len:
                #lbatch, tmpinfo = self.process_batch(lbatch)
                #info += tmpinfo
                # TODO : it's dirty to skip the features !
                classif_ret = self.classifier(lbatch_img[:self.batch_len])[1:]
                for i in range(self.batch_len):
                    # TODO: its dirty to put face confidence at the end, its for test retro compatibility
                    info.append(lbatch_info[i] + [e[i] for e in classif_ret])
                lbatch_img = lbatch_img[self.batch_len:]
                lbatch_info = lbatch_info[self.batch_len:]

        if len(lbatch_img) > 0:
            #lbatch, tmpinfo = self.process_batch(lbatch)
            #info += tmpinfo
            classif_ret = self.classifier(lbatch_img)[1:]
            for i in range(len(lbatch_img)):
                info.append(lbatch_info[i] + [e[i] for e in classif_ret])
        return pd.DataFrame.from_records(info, columns = ['frame', 'bb', 'face_detect_conf'] + self.classifier.outnames[1:])

    def pred_from_vid_and_bblist(self, vidsrc, lbox, subsamp_coeff=1, start_frame=0):
        lret = []
        lfeat = []

        for (iframe, frame), bbox in zip(video_iterator(vidsrc, subsamp_coeff=subsamp_coeff, start=start_frame, verbose=self.verbose),lbox):
            if self.verbose:
                print('iframe: %s, bbox: %s' % (iframe, bbox))

            ret = self.classif_from_frame_and_bbox(frame, bbox, self.squarify_bbox, self.bbox_scaling, True)

            lret.append(ret[1:])
            lfeat.append(ret[0])

            if self.verbose:
                print('bounding box (x1, y1, x2, y2),' + ','.join(self.classifier.outnames[1:]))
                print(lret[-1])
                print()
        assert len(lret) == len(lbox), '%d bounding box provided, and only %d frames processed' % (len(lbox), len(lret))
        return np.concatenate(lfeat), pd.DataFrame.from_records(lret, columns=['bb']+ self.classifier.outnames[1:])


# TODO : kwarfs for providing arguments to super class ??
# use super name also !
class GenderTracking(AbstractGender):
    def __init__(self, detection_period, face_detector = OcvCnnFacedetector(paddpercent=0.), face_classifier=Vggface_LSVM_YTF(), bbox_scaling=1.1, squarify=True, verbose = False):
        AbstractGender.__init__(self, face_detector, face_classifier, bbox_scaling, squarify, verbose)
        self.detection_period = detection_period

    def __call__(self, video_path, subsamp_coeff = 1 ,  offset = -1):

        """
        Pipeline function for gender classification from videos without tracking.

        Parameters:
            video_path (string): Path for input video.
            subsamp_coeff (int) : only 1/subsamp_coeff frames will be processed
            offset (float) : Time in milliseconds to skip at the beginning of the video.


        Returns:
            info: A Dataframe with frame and face information (coordinates, decision function,labels..)
        """


        oshape = self.classifier.input_shape[:-1]
        #classif_desc = ','.join(self.classifier.outnames[1:])
        detector = TrackerDetector(self.face_detector, self.detection_period)

        info = []
        lbatch_img = []
        lbatch_info = []

        for iframe, frame in video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=min(offset, 0), verbose=self.verbose):

            for bb, faceid, detect_conf, track_conf in detector(frame):
                if self.verbose:
                    print(detector.out_names, bb, faceid, detect_conf, track_conf)

                face_img, bbox = preprocess_face(frame, bb, self.squarify_bbox, self.bbox_scaling, True, self.face_alignment, oshape, self.verbose)

                #lbatch.append((iframe, bb, detect_conf, face_img, bbox))
                lbatch_info.append([iframe, bbox, faceid, detect_conf, track_conf])
                lbatch_img.append(face_img)

            while len(lbatch_img) > self.batch_len:
                # TODO : it's dirty to skip the features !
                classif_ret = self.classifier(lbatch_img[:self.batch_len])[1:]
                for i in range(self.batch_len):
                    info.append(lbatch_info[i] + [e[i] for e in classif_ret])
                lbatch_img = lbatch_img[self.batch_len:]
                lbatch_info = lbatch_info[self.batch_len:]

        if len(lbatch_img) > 0:
            classif_ret = self.classifier(lbatch_img)[1:]
            for i in range(len(lbatch_img)):
                info.append(lbatch_info[i] + [e[i] for e in classif_ret])
        df = pd.DataFrame.from_records(info, columns = ['frame', 'bb'] + detector.out_names[1:]+ self.classifier.outnames[1:])
        return _smooth_labels(df)


# should be moved in face_classifier
# face classifiers should implement their own smoothing procedure and
# define the fields requiring smoothing
def _label_decision_fun(x):

    if x>0:
        return 'm'
    else:
        return 'f'

def _smooth_labels(df):
    if len(df) == 0:
        df['avg_sex_decision_function'] = []
        df['avg_sex_label'] = []
        return df

    byfaceid = pd.DataFrame(df.groupby('face_id')['sex_decision_function'].mean())
    byfaceid.rename(columns = {'sex_decision_function':'avg_sex_decision_function'}, inplace=True)
    new_df = df.merge(byfaceid, on= 'face_id')
    new_df['avg_sex_label'] = new_df['avg_sex_decision_function'].map(_label_decision_fun)

    # I guess with a different joining/merging strategy, there would be no need to sort
    return new_df.sort_values(by = ['frame', 'face_id']).reset_index(drop=True)
