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
import os
import h5py
from sklearn.svm import LinearSVC

from .face_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from .opencv_utils import video_iterator
from .face_tracking import TrackerList
from .face_detector import OcvCnnFacedetector

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image 


from matplotlib import pyplot as plt

    

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

    

  
    
  
    
class GenderVideo:
    """ 
    This is a class regrouping all phases of a pipeline designed for gender classification from video.
    
    Attributes: 
        face_detector: Face detection model. 
        align_predictor: Face alignment model.
        gender_svm: Gender SVM classifier model.
        vgg_feature_extractor: VGGFace neural model used for feature extraction.
        threshold: quality of face detection considered acceptable, value between 0 and 1.
    """
    def __init__(self, face_detector = OcvCnnFacedetector(bbox_scaling=1.1), verbose = False):
        
        """ 
        The constructor for GenderVideo class. 
  
        Parameters: 
           threshold (float): quality of face detection considered acceptable, value between 0 and 1. 
        """
        
        p = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        #self.face_detector = cv2.dnn.readNetFromTensorflow(p + "opencv_face_detector_uint8.pb",
        #                                                   p + "opencv_face_detector.pbtxt")
        self.face_detector = face_detector
        self.align_predictor = dlib.shape_predictor(p +'shape_predictor_68_face_landmarks.dat')

        
        f = h5py.File(p + 'svm_classifier.hdf5', 'r')
        svm = LinearSVC()
        svm.classes_ = np.array(f['linearsvc/classes'][:]).astype('<U1')
        svm.intercept_ = f['linearsvc/intercept'][:]
        svm.coef_ = f['linearsvc/coef'][:]
        self.gender_svm = svm
        
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = (224, 224, 3), pooling ='avg')
        self.verbose = verbose

    def _gender_from_face(self, img):
        """
        Face is supposed to be aligned and cropped and resized to 224*224
        it is for regulard detection __call__
        we should check if it is done in the tracking implementation
        """
        img = image.img_to_array(img)
        img = utils.preprocess_input(img, version=1)
        img = np.expand_dims(img, axis=0)
        features = self.vgg_feature_extractor.predict(img)
        label = self.gender_svm.predict(features)[0]
        decision_value = round(self.gender_svm.decision_function(features)[0], 3)
        return label, decision_value

    
    def align_and_crop_face(self, img, bb, desired_width, desired_height):
        """ 
        Aligns and resizes face to desired shape.
  
        Parameters: 
            img  : Image to be aligned and resized.
            bb: Bounding box coordinates tuples. (dlib.rectangle)
            desired_width: output image width.
            desired_height: output image height.
            
          
        Returns: 
            cropped_img: Image aligned and resized.
            left_eye: left eye position coordinates.
            right_eye: right eye position coordinates.
        """        
        shape = self.align_predictor(img, bb)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)
        M = get_rotation_matrix(left_eye, right_eye)

        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        cropped = crop_image(rotated_img, bb)
        cropped_res = cv2.resize(cropped, (desired_width, desired_height))
        cropped_img = cropped_res[:, :, ::-1]

        return cropped_img, left_eye, right_eye
        



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
                label, decision_value = self._gender_from_face(face_img)
                
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

            faces_info = self.face_detector(frame)

            for bb, detect_conf in faces_info:
                face_img, left_eye, right_eye = self.align_and_crop_face(frame, dlib.rectangle(*bb), 224, 224)
                label, decision_value = self._gender_from_face(face_img)

                info.append([iframe, bb, label, decision_value, round(detect_conf, 3)])

        info = pd.DataFrame.from_records(info, columns = ['frame', 'bb','label', 'decision', 'conf'])
        return info
    

    def pred_from_frame_and_bb(self, frame, bb, scale=1, display=False): ## add bounding box scaling
        # frame should be obtained from opencv
        # bounding box is x1, y1, x2, y2

        if display:
            plt.imshow(frame)
            plt.show()

        x1, y1, x2, y2 = [e for e in bb]
        # TODO: check the result without the frame.shape argument
        x1, y1, x2, y2 = _scale_bbox(x1, y1, x2, y2, scale, frame.shape)

        dets = [dlib.rectangle(x1, y1, x2, y2)]

        if display:
            plt.imshow(frame[y1:y2, x1:x2, :])
            plt.show()

        frame = self.align_and_crop_face(frame, dets, 224, 224)[0]
        if display:
            plt.imshow(frame)
            plt.show()

        ret = self._gender_from_face(frame)
        if display:
            print(ret)
        return ret

    def pred_from_vid_and_bblist(self, vidsrc, lbox, subsamp_coeff=1, start_frame=0, scale=1., display=False):
        lret = []
        for (iframe, frame), bbox in zip(video_iterator(vidsrc, subsamp_coeff=subsamp_coeff, start=start_frame),lbox):
            lret.append(self.pred_from_frame_and_bb(frame, bbox, scale, display))
        return lret


