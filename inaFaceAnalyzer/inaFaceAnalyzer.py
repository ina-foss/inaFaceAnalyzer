#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2021 Ina (David Doukhan & Zohra Rezgui - http://www.ina.fr/)

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

import pandas as pd
from abc import ABC, abstractmethod
from .opencv_utils import video_iterator, image_iterator, analysisFPS2subsamp_coeff
from .face_tracking import TrackerDetector
from .face_detector import OcvCnnFacedetector, PrecomputedDetector
from .face_classifier import Resnet50FairFaceGRA
from .face_alignment import Dlib68FaceAlignment
from .face_preprocessing import preprocess_face


class FaceAnalyzer(ABC):
    """
    This is an abstract class containg the common code to be used to process
    images, videos, with/without tracking
    """
    batch_len = 32

    def __init__(self, face_detector = None, face_classifier = None, bbox_scaling = 1.1, squarify_bbox = True, verbose = False):
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
        if face_detector is None:
            self.face_detector = OcvCnnFacedetector(padd_prct=0.)
        else:
            self.face_detector = face_detector


        # set all bounding box shapes to square
        self.squarify_bbox = squarify_bbox

        # scaling factor to be applied to face bounding boxes
        self.bbox_scaling = bbox_scaling

        # face alignment module
        self.face_alignment = Dlib68FaceAlignment()

        # Face feature extractor from aligned and detected faces
        if face_classifier is None:
            self.classifier = Resnet50FairFaceGRA()
        else:
            self.classifier = face_classifier

        # True if some verbose is required
        self.verbose = verbose


    @abstractmethod
    def __call__(self, src) : pass

    def _process_stream(self, stream_iterator, detector):
        oshape = self.classifier.input_shape[:-1]

        lbatch_img = []
        linfo = []
        ldf = []

        for iframe, frame in stream_iterator:

            for detection in detector(frame, self.verbose):
                if self.verbose:
                    print(detection)

                face_img, bbox = preprocess_face(frame, detection, self.squarify_bbox, self.bbox_scaling, self.face_alignment, oshape, self.verbose)

                linfo.append([iframe, detection._replace(bbox=tuple(bbox))])
                lbatch_img.append(face_img)

            while len(lbatch_img) > self.batch_len:
                df = self.classifier(lbatch_img[:self.batch_len], False)
                ldf.append(df)
                lbatch_img = lbatch_img[self.batch_len:]

        if len(lbatch_img) > 0:
            df = self.classifier(lbatch_img, False)
            ldf.append(df)

        if len(ldf) == 0:
            return pd.DataFrame(None, columns=(['frame'] + list(detector.output_type._fields) + self.classifier.output_cols))

        df1 = pd.DataFrame({'frame' : [e[0] for e in linfo]})
        df2 = pd.DataFrame.from_records([e[1] for e in linfo], columns=detector.output_type._fields)
        if 'eyes' in df2.columns:
            df2 = df2.drop('eyes', axis=1)
        df3 = pd.concat(ldf).reset_index(drop=True)
        return pd.concat([df1, df2, df3], axis = 1)


class ImageAnalyzer(FaceAnalyzer):

    def __init__(self, **kwargs):
        if 'face_detector' not in kwargs:
            kwargs['face_detector'] = OcvCnnFacedetector()
        super().__init__(**kwargs)

    def __call__(self, img_paths):
        if isinstance(img_paths, str):
            stream = image_iterator([img_paths], verbose = self.verbose)
        else:
            stream = image_iterator(img_paths, verbose = self.verbose)
        return self._process_stream(stream, self.face_detector)

class VideoAnalyzer(FaceAnalyzer):
    """
    This is a class regrouping all phases of a pipeline designed for gender classification from video.

    Attributes:
        face_detector: Face detection model.
        face_alignment: Face alignment model.
        gender_svm: Gender SVM classifier model.
        vgg_feature_extractor: VGGFace neural model used for feature extraction.
        threshold: quality of face detection considered acceptable, value between 0 and 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, video_path, fps = None,  offset = 0):

        """
        Pipeline function for gender classification from videos without tracking.

        Parameters:
            video_path (string): Path for input video.
            subsamp_coeff (int) : only 1/subsamp_coeff frames will be processed
            offset (float) : Time in milliseconds to skip at the beginning of the video.


        Returns:
            info: A Dataframe with frame and face information (coordinates, decision function,labels..)
        """
        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=max(offset, 0), verbose=self.verbose)
        return self._process_stream(stream, self.face_detector)


class VideoPrecomputedDetection(VideoAnalyzer):
    def __init__(self, **kwargs):
        if 'face_detector' in kwargs:
            raise NotImplementedError('VideoPrecomputedDetection should NOT be constructed with a face detector')
        kwargs['face_detector'] = PrecomputedDetector()
        super().__init__(**kwargs)
    def __call__(self, video_path, lbbox, fps=None, start_frame = 0):
        detector = PrecomputedDetector(lbbox)

        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='frame', start=start_frame, verbose=self.verbose)
        df = self._process_stream(stream, detector)

        assert len(detector.lbbox) == 0, 'the detection list is longer than the number of processed frames'
        return df

class VideoTracking(VideoAnalyzer):
    def __init__(self, detection_period, **kwargs):
        super().__init__(**kwargs)
        self.detection_period = detection_period

    def __call__(self, video_path, fps = None,  offset = -1):

        """
        Pipeline function for gender classification from videos without tracking.

        Parameters:
            video_path (string): Path for input video.
            subsamp_coeff (int) : only 1/subsamp_coeff frames will be processed
            offset (float) : Time in milliseconds to skip at the beginning of the video.


        Returns:
            info: A Dataframe with frame and face information (coordinates, decision function,labels..)
        """

        detector = TrackerDetector(self.face_detector, self.detection_period)

        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=max(offset, 0), verbose=self.verbose)

        df = self._process_stream(stream, detector)

        return self.classifier.average_results(df)
