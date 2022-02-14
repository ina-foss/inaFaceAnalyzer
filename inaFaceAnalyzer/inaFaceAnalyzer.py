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
from .pyav_utils import video_keyframes_iterator
from .face_tracking import TrackerDetector
from .face_detector import LibFaceDetection, PrecomputedDetector
from .face_classifier import Resnet50FairFaceGRA
from .face_alignment import Dlib68FaceAlignment
from .face_preprocessing import preprocess_face


class FaceAnalyzer(ABC):
    """
    This is an abstract class containg the pipeline used to process
    images, videos, with/without tracking
    * image/video decoding
    * face detection
    * face tracking (optional)
    * eye detection
    * face preprocessing
    * face classification
    """

    # len of batches to be sent to face classifiers
    #batch_len = 32

    def __init__(self, face_detector = None, face_classifier = None, batch_len=32, verbose = False):
        """
        Constructor
        Parameters
        ----------
        face_detector : instance of face_detector.FaceDetector or None
            if None, LibFaceDetection is used by default
        face_classifier: instance of face_classifier.FaceClassifier or None
            if None, Resnet50FairFaceGRA is used by default (gender & age)
        verbose : boolean
            If True, will display several usefull intermediate images and results
        """

        # face detection system
        if face_detector is None:
            face_detector = LibFaceDetection()
        self.face_detector = face_detector

        # Face feature extractor from aligned and detected faces
        if face_classifier is None:
            face_classifier = Resnet50FairFaceGRA()
        self.classifier = face_classifier

        # if set to True, then the bounding box (manual or automatic) is set
        # to the smallest square containing the bounding box
        self.bbox2square = face_classifier.bbox2square

        # scaling factor to be applied to the face bounding box after detection
        # larger bounding box may help for sex classification from face
        self.bbox_scale = face_classifier.bbox_scale

        # face alignment module
        self.face_alignment = Dlib68FaceAlignment()

        # True if some verbose is required
        assert isinstance(verbose, bool)
        self.verbose = verbose

        # set to large values with large memory GPU for faster processing times !
        assert isinstance(batch_len, int) and batch_len > 0
        self.batch_len = batch_len


    @abstractmethod
    def __call__(self, src) :
        """
        Method to be implemented by each analyzer

        Parameters
        ----------
        src : str or list
            path to the video/image to be analyzed
            May also be a list of images
        Returns
        -------
        Results stored in a pandas DataFrame
        """
        pass

    def _process_stream(self, stream_iterator, detector):
        """
        Generic pipeline allowing to process image or video streams
        Faces are first detected, preprocessed and sent in batches in
        face classifiers

        Parameters
        ----------
        stream_iterator : iterator
            iterator returning decoded RBG images at each call together with
            an image identifier
            see: opencv_utils.video_iterator, opencv_utils.image_iterator,
            pyav_utils.video_keyframes_iterator
        detector : instance of face_tracking, tracker_detector or face_detector.FaceDetector

        Returns
        -------
        TYPE
            pandas Dataframe with analysis results

        """
        oshape = self.classifier.input_shape[:-1]

        lbatch_img = []
        linfo = []
        ldf = []

        # iterate on image list or video stream
        for iframe, frame in stream_iterator:

            # iterate on detected faces
            for detection in detector(frame, self.verbose):
                if self.verbose:
                    print(detection)

                # preprocess detected faces: bbox normalization, eye detection, rotation, ...
                face_img, bbox = preprocess_face(frame, detection, self.bbox2square, self.bbox_scale, self.face_alignment, oshape, self.verbose)

                linfo.append([iframe, detection._replace(bbox=tuple(bbox))])
                # store preprocessed faces in a list, for further batch processing
                lbatch_img.append(face_img)

            # if enough faces were found, process a batch of faces
            while len(lbatch_img) > self.batch_len:
                df = self.classifier(lbatch_img[:self.batch_len], False)
                ldf.append(df)
                lbatch_img = lbatch_img[self.batch_len:]


        if len(lbatch_img) > 0:
            df = self.classifier(lbatch_img, False)
            ldf.append(df)

        if len(ldf) == 0:
            return pd.DataFrame(None, columns=(['frame'] + list(detector.output_type._fields) + self.classifier.output_cols))

        # return results as a pandas Dataframe
        df1 = pd.DataFrame({'frame' : [e[0] for e in linfo]})
        df2 = pd.DataFrame.from_records([e[1] for e in linfo], columns=detector.output_type._fields)
        if 'eyes' in df2.columns:
            df2 = df2.drop('eyes', axis=1)
        df3 = pd.concat(ldf).reset_index(drop=True)
        return pd.concat([df1, df2, df3], axis = 1)


class ImageAnalyzer(FaceAnalyzer):
    """
    ImageAnalyzer instances allow to detect and classify faces from images
    """
    def __call__(self, img_paths):
        """
        Parameters
        ----------
        img_paths : str or list
            path or list of paths to image file(s) to analyze
        Returns
        -------
        pandas Dataframe with column 'frame' containing the path to the source
        image. Remaining columns depend on processing options selected and
        contain bounding box, and face classification information

        """
        if isinstance(img_paths, str):
            stream = image_iterator([img_paths], verbose = self.verbose)
        else:
            stream = image_iterator(img_paths, verbose = self.verbose)
        return self._process_stream(stream, self.face_detector)


class VideoAnalyzer(FaceAnalyzer):
    """
    Video Analyzer allows to detect and classify faces in video streams
    """
    def __call__(self, video_path, fps = None,  offset = 0):
        """
        Pipeline function for face classification from videos (without tracking)

        Parameters:
            video_path: str
                Path to input video.
            fps: float or None (default)
                amount of video frames to process per seconds
                if set to None, all frames are processed (costly)
            offset: float (default: 0)
                Time in milliseconds to skip at the beginning of the video.

        Returns:
            Dataframe with frame and face information: frame position,
            coordinates, predictions, decision function,labels...
        """
        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=max(offset, 0), verbose=self.verbose)
        return self._process_stream(stream, self.face_detector)


class VideoKeyframes(FaceAnalyzer):
    """
    Face detection and analysis from video limited to video key frames
    https://en.wikipedia.org/wiki/Key_frame
    It allows to provide a video analysis summary in fast processing time, but
    with non uniform frame sampling rate
    """
    def __call__(self, video_path):

        """
        Pipeline function for face classification from videos, limited to key frames

        Parameters:
            video_path (string): Path for input video.
        Returns:
            Dataframe with frame and face information: frame position,
            coordinates, predictions, decision function,labels...
        """
        stream = video_keyframes_iterator(video_path, verbose=self.verbose)
        return self._process_stream(stream, self.face_detector)


class VideoTracking(FaceAnalyzer):
    """
    Video processing pipeline including face detection, tracking and classification
    Tracking is usually less costly than face detection (computation bottleneck)
    and allows to save computation time
    Classification decision functions and predictions are averaged for each
    tracked faces, allowing to obtain more robust analysis estimates
    """
    def __init__(self, detection_period, face_detector = None, face_classifier = None, batch_len=32, verbose = False):
        """
        Constructor

        Parameters
        ----------
        detection_period : int
            the face detection algorithm (costly) will be used once every
            'detection_period' analyzed frames.
            Ie: if set to 5, face detection will occur for 1/5 frames and the
            remaining 4/5 faces will be detected through a tracking procedure
                if set to 1: face detection will occur for each frame. Face
                tracking will also be used for each frames, since it will allow
                to group same faces under a person identifier
        face_detector : instance of face_detector.FaceDetector or None, optional
            if None, LibFaceDetection is used. The default is None.
        face_classifier : instance of face_classifier.FaceClassifier or None, optional
            if None, Resnet50FairFaceGRA is used (gender & age). The default is None.
        verbose : boolean, optional
            If True, will display several usefull intermediate images and results.
            The default is False.
        """
        super().__init__(face_detector, face_classifier, batch_len=batch_len, verbose=verbose)
        self.detection_period = detection_period

    def __call__(self, video_path, fps = None,  offset = 0):
        """
        Pipeline function for face classification from videos with tracking

        Parameters:
            video_path: str
                Path to input video.
            fps: float or None (default)
                amount of video frames to process per seconds
                if set to None, all frames are processed (costly)
            offset: float (default: 0)
                Time in milliseconds to skip at the beginning of the video.

        Returns:
            Dataframe with frame and face information: frame position,
            coordinates, predictions, decision function,labels...
            faceid column allow to keep track of each unique face found
            predictions and decision functions with '_avg' suffix are obtained
            through a smoothing procedure of decision functions for all faces
            with same faceid. Smoothed estimates are usually more robust than
            instantaneous ones
        """
        detector = TrackerDetector(self.face_detector, self.detection_period)

        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=max(offset, 0), verbose=self.verbose)

        df = self._process_stream(stream, detector)

        return self.classifier.average_results(df)



class VideoPrecomputedDetection(FaceAnalyzer):
    """
    Video analysis class to be used combined with pre-detected face bounding
    boxes (uncommon use-case)
    CANNOT BE USED WITH RESCALED OR SQUARIFIED BOUNDING BOXES!!!
    FACE ALIGNMENT REQUIRE TO HAVE ORIGINAL BOUNDING BOX TO PERFORM WELL!!
    """
    def __init__(self, face_classifier = None, verbose = False, bbox_scale=None, bbox2square=None):
        """
        Constructor

        Parameters
        ----------
        face_classifier: instance of face_classifier.FaceClassifier or None
            if None, Resnet50FairFaceGRA is used by default (gender & age)
        verbose : boolean
            If True, will display several usefull intermediate images and results
        bbox_scale: float or None
            scaling factor to be applied to the face bounding box after detection
            if not None, will overrides classifier's bbox scaling instructions
            usefull if provided bounding boxes are already transformed - default None
        bbox2square: boolean or None
            set bounding box to the smallest square containing the bounding box
            if not None, will ovverides classifier's bbox2square instructions
            usefull if provided bounding boxes are already transformed - default None
        """

        super().__init__(PrecomputedDetector(), face_classifier, verbose=verbose)
        if bbox_scale is not None:
            self.bbox_scale = bbox_scale
        if bbox2square is not None:
            self.bbox2square = bbox2square

    def __call__(self, video_path, lbbox, fps=None, start_frame = 0):
        """
        Pipeline function for face classification from videos using pre-detected faces

        Parameters:
            video_path: str
                Path to input video.
            lbbox: list of bounding boxes
                Each list element i contain either a tuple or a list of tuples
                (x1,y1,x2,y2) corresponding to the face found in the ith frame
            fps: float or None (default)
                amount of video frames to process per seconds
                if set to None, all frames are processed (costly)
            offset: float (default: 0)
                Time in milliseconds to skip at the beginning of the video.

        Returns:
            Dataframe with frame and face information: frame position,
            coordinates, predictions, decision function,labels...
        """
        detector = PrecomputedDetector(lbbox)

        subsamp_coeff = 1 if fps is None else analysisFPS2subsamp_coeff(video_path, fps)
        stream = video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='frame', start=start_frame, verbose=self.verbose)
        df = self._process_stream(stream, detector)

        assert len(detector.lbbox) == 0, 'the detection list is longer than the number of processed frames'
        return df
