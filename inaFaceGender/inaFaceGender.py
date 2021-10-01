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

import dlib, cv2
import numpy as np
import pandas as pd
from .opencv_utils import video_iterator
from .face_tracking import TrackerList
from .face_detector import OcvCnnFacedetector
from .face_classifier import VGG16_LinSVM
from .face_alignment import Dlib68FaceAlignment


from matplotlib import pyplot as plt


# def crop_image_fill(frame, bb):
#     x1, y1, x2, y2 = bb
#     hframe, wframe = (frame.shape[0], frame.shape[1])
#     if x1 >= 0 and y1 >= 0 and x2 <= wframe and y2 <= hframe:
#         return frame[top:bottom, left:right]
#     w = x2 - x1
#     h = y2 - y1
#     sframe = frame[max(0, y1):min(y2, hframe), max(0, x1):min(x2, wframe),:]
#     ret = np.zeros((h, w, 3), np.uint8)
#     yoff = (h - sframe.shape[0]) // 2
#     xoff = (w - sframe.shape[1]) // 2
#     print('xoff', xoff, 'yoff', yoff)
#     ret[yoff:(h+yoff), xoff:(w+xoff), :] = sframe[:,:,:]
#     return ret



def info2csv(df, csv_path):
    """
    Write df into a csv.


    Parameters:
    df (DataFrame): Dataframe to be written to csv.
    csv_path (string): CSV output path.

    """
    df.to_csv(csv_path, index=False)

def _label_decision_fun(x):

    if x>0:
        return 'm'
    else:
        return 'f'

def _smooth_labels(df):
    if len(df) == 0:
        df['smoothed_decision'] = []
        df['smoothed_label'] = []
        return df

    byfaceid = pd.DataFrame(df.groupby('faceid')['decision'].mean())
    byfaceid.rename(columns = {'decision':'smoothed_decision'}, inplace=True)
    new_df = df.merge(byfaceid, on= 'faceid')
    new_df['smoothed_label'] = new_df['smoothed_decision'].map(_label_decision_fun)

    return new_df

def _scale_bbox(x1, y1, x2, y2, scale):
    w = x2 - x1
    h = y2 - y1

    x1 = int(x1 - (w*scale - w)/2)
    y1 = int(y1 - (h*scale -h)/2)
    x2 = int(x2 + (w*scale - w)/2)
    y2 = int(y2 + (h*scale -h)/2)

    # if frame_shape is not None:
    #     x1 = max(x1, 0)
    #     y1 = max(y1, 0)
    #     x2 = min(x2, frame_shape[1])
    #     y2 = min(y2, frame_shape[0])

    return x1, y1, x2, y2

def _squarify_bbox(bbox):
    """
    Convert a rectangle bounding box to a square
    width sides equal to the maximum between width and height
    """
    #print('sq' , bbox)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    offset = max(w, h) / 2
    x_center = (x1+x2) / 2
    y_center = (y1+y2) / 2
    return x_center - offset, y_center - offset, x_center + offset, y_center + offset

# TODO: this may cause a stretching in a latter stage
# TODO conversion to int should be done after scaling
def _norm_bbox(bbox, frame_width, frame_height):
    """
    convert to int and crop bbox to 0:frame_shape
    """
    x1, y1, x2, y2 = [int(e) for e in bbox]
    return max(0, x1), max(0, y1), min(x2, frame_width), min(y2, frame_height)



class AbstractGender:
    """
    This is an abstract class containg the common code to be used to process
    images, videos, with/without tracking
    """
    # TODO : add squarify bbox option ???
    def __init__(self, face_detector, bbox_scaling, squarify_bbox, verbose):
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
        #p = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        # face detection system
        self.face_detector = face_detector

        # set all bounding box shapes to square
        self.squarify_bbox = squarify_bbox

        # scaling factor to be applied to face bounding boxes
        self.bbox_scaling = bbox_scaling

        # face alignment module
        # TODO: make a separate class
#        self.align_predictor = dlib.shape_predictor(p +'shape_predictor_68_face_landmarks.dat')
        self.face_alignment = Dlib68FaceAlignment(verbose=verbose)

        # Face feature extractor from aligned and detected faces
        self.classifier = VGG16_LinSVM()

        # True if some verbose is required
        self.verbose = verbose



    def preprocess_face(self, frame, bbox, squarify, bbox_scale, norm, align_eyes, output_shape, verbose=False):
        """
        Apply preprocessing pipeline to a detected face and returns the
        corresponding image with the following optional processings
        Parameters
        ----------
        frame : numpy nd.array
            RGB image data
        bbox : tuple (x1, y1, x2, y2) or None
            if not None, the provided bounding box will be used, else the
            whole image will be used
        squarify: boolean
            if set to True, the bounding box will be extented to be the
            smallest square containing the bounding box. This avoids distortion
            if faces are resized in a latter stage
        bbox_scale: float
            resize the bounding box to consider larger (bbox_scale > 1) or
            smaller (bbox_scale < 1) areas around faces. If set to 1, the
            original bounding box is used
        norm : boolean
            if set to True, bounding boxes will be converted from float to int
            and will be cropped to fit inside the input frame
        align_eyes: boolean
            if set to True, face will be rotated based on the estimation of
            facial landmarks such the eyes lie on a horizontal line
        output_shape: (width, height) or None
            if not None, face will be resized to the provided output shape
        Returns
        -------
        frame: np.array RGB image data
        bbox: up-to-data bounding box after the proprocessing pipeline

        """

        (frame_h, frame_w, _) = frame.shape

        # if no bounding box is provided, use the whole image
        if bbox is None:
            bbox = (0, 0, frame_w, frame_h)

        # if True, extend the bounding box to the smallest square containing
        # the orignal bounding box
        if squarify:
            bbox = _squarify_bbox(bbox)

        # perform bounding box scaling to march larger/smaller areas around
        # the detected face
        bbox = _scale_bbox(*bbox, bbox_scale)

        # bounding box normalization to int and to fit in input frame
        if norm:
            bbox = _norm_bbox(bbox, frame_w, frame_h)

        # if verbose, display the image and the bounding box
        if verbose:
           tmpframe = frame.copy()
           cv2.rectangle(tmpframe, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 8)
           plt.imshow(tmpframe)
           plt.show()

        # performs face alignment based on facial landmark detection
        if align_eyes:
            frame, left_eye, right_eye = self.face_alignment(frame, bbox)

        # crop image to the bounding box
        frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # resize image to the required output shape
        if output_shape is not None:
            frame = cv2.resize(frame, output_shape)

        if verbose:
            print('resulting image')
            plt.imshow(frame)
            plt.show()

        return frame, bbox

    def classif_from_frame_and_bbox(self, frame, bbox, bbox_square, bbox_scale, bbox_norm):

        face_img, bbox = self.preprocess_face(frame, bbox, bbox_square, bbox_scale, bbox_norm, True, (224, 224), self.verbose)

        feats, label, decision_value = self.classifier(face_img)
        ret = [feats, bbox, label, decision_value]

        return ret

    def detect_and_classify_faces_from_frame(self, frame):
        if self.verbose:
            plt.imshow(frame)
            plt.show()
        ret = []
        for bb, detect_conf in self.face_detector(frame):
            if self.verbose:
                print('bbox: %s, conf: %f' % (bb, detect_conf))
            ret.append(self.classif_from_frame_and_bbox(frame, bb, self.squarify_bbox, self.bbox_scaling, True) + [detect_conf])

            if self.verbose:
                print('bounding box (x1, y1, x2, y2), sex label, sex classification decision function, face detection confidence')
                print(ret[-1][1:])
                print()
        return ret



class GenderImage(AbstractGender):
    def __init__(self, face_detector = OcvCnnFacedetector(), bbox_scaling=1.1, squarify=True, verbose = False):
        AbstractGender.__init__(self, face_detector, bbox_scaling, squarify, verbose)

    def read_rgb(self, img_path, verbose=False):
        img = cv2.imread(img_path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if verbose:
            print('raw image ' + img_path)
            plt.imshow(frame)
            plt.show()
        return frame

    def __call__(self, img_path):
        frame = self.read_rgb(img_path, self.verbose)
        return self.detect_and_classify_faces_from_frame(frame)
    # def get_face_images(self, img_path):
    #     img_cv2.imread(img_path)
    #     frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    def __init__(self, face_detector = OcvCnnFacedetector(), bbox_scaling=1.1, squarify=True, verbose = False):
        AbstractGender.__init__(self, face_detector, bbox_scaling, squarify, verbose)

    # TODO: BUILD A SEPARATE CLASS FOR DETECTION+TRACKING
    def detect_with_tracking(self, video_path, k_frames, subsamp_coeff = 1, offset = -1):
        """
        Pipeline for gender classification from videos using correlation filters based tracking (dlib's).

        Parameters:
            video_path (string): Path for input video.
            k_frames (int) : Number of frames for which continue tracking the faces without renewing face detection.
            subsamp_coeff (int) : only 1/subsamp_coeff frames will be processed
            offset (float) : Time in milliseconds to skip at the beginning of the video.

        Returns:
            info (DataFrame): A Dataframe with frame and face information (coordinates, decision function, smoothed and non smoothed labels)
        """

        assert (k_frames % subsamp_coeff) == 0


        tl = TrackerList()

        info = []

        for iframe, frame in video_iterator(video_path,subsamp_coeff=subsamp_coeff, time_unit='ms', start=min(offset, 0)):

            tl.update(frame)

            # detect faces every k frames
            if (iframe % k_frames)==0:
                faces_info = self.face_detector(frame)

                tl.ingest_detection(frame, [dlib.rectangle(*e[0]) for e in faces_info])

            # process faces based on position found in trackers
            for fid in tl.d:
                bb = tl.d[fid].t.get_position()

                x1, y1, x2, y2 = _scale_bbox(bb.left(), bb.top(), bb.right(), bb.bottom(), 1, frame.shape)
                if x1 < x2 and y1 < y2:
                    bb = dlib.rectangle(x1, y1, x2, y2)
                else:
                    ## TODO WARNING - THIS HACK IS STRANGE
                    bb = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])


                face_img, left_eye, right_eye = self.align_and_crop_face(frame, bb, 224, 224)
                #label, decision_value = self._gender_from_face(face_img)
                label, decision_value = self.classifier(face_img)

                bb =  (bb.left(), bb.top(), bb.right(), bb.bottom())

                # TODO - ADD CONFIDENCE
                info.append([
                        iframe, fid,  bb, label,
                        decision_value # confidence[fid]
                    ])



        track_res = pd.DataFrame.from_records(info, columns = ['frame', 'faceid', 'bb','label', 'decision']) #, 'conf'])
        info = _smooth_labels(track_res)

        return info


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

        info = []

        for iframe, frame in video_iterator(video_path, subsamp_coeff=subsamp_coeff, time_unit='ms', start=min(offset, 0)):

            info += [[iframe] + e[1:] for e in self.detect_and_classify_faces_from_frame(frame)]

        info = pd.DataFrame.from_records(info, columns = ['frame', 'bb','label', 'decision', 'conf'])
        return info



    def pred_from_vid_and_bblist(self, vidsrc, lbox, subsamp_coeff=1, start_frame=0):
        lret = []
        lfeat = []

        for (iframe, frame), bbox in zip(video_iterator(vidsrc, subsamp_coeff=subsamp_coeff, start=start_frame),lbox):
            if self.verbose:
                print('iframe: %s, bbox: %s' % (iframe, bbox))

            ret = self.classif_from_frame_and_bbox(frame, bbox, self.squarify_bbox, self.bbox_scaling, True)

            lret.append(ret[1:])
            lfeat.append(ret[0])

            if self.verbose:
                print('bounding box (x1, y1, x2, y2), sex label, sex classification decision function')
                print(lret[-1])
                print()
        assert len(lret) == len(lbox), '%d bounding box provided, and only %d frames processed' % (len(lbox), len(lret))
        return np.concatenate(lfeat), pd.DataFrame.from_records(lret, columns=['bb', 'label', 'decision'])
