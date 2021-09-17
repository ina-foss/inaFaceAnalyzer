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

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image 


from matplotlib import pyplot as plt

    
def _get_bbox_pts(detections, face_idx, frame_width, frame_height):
    # TODO: refactor
    # this code is 100% dependent on the face detection method used
    # build a face detection class ???
    
    x1 = int(detections[0, 0, face_idx, 3] * frame_width)
    y1 = int(detections[0, 0, face_idx, 4] * frame_height)
    x2 = int(detections[0, 0, face_idx, 5] * frame_width)
    y2 = int(detections[0, 0, face_idx, 6] * frame_height)

    # TODO: check here if x1 < x2 ???

    width = x2 - x1
    height = y2 - y1
    max_size = max(width, height)
    x1, x2 = max(0, (x1 + x2) // 2 - max_size // 2), min(frame_width, (x1 + x2) // 2 + max_size // 2)
    y1, y2 = max(0, (y1 + y2) // 2 - max_size // 2), min(frame_height, (y1 + y2) // 2 + max_size // 2)

    return x1, y1, x2, y2


def _scale_bbox(x1, y1, x2, y2, scale, frame_shape=None):
    w = x2 - x1
    h = y2 - y1
    
    x1 = int(x1 - (w*scale - w)/2)
    y1 = int(y1 - (h*scale -h)/2)
    x2 = int(x2 + (w*scale - w)/2)
    y2 = int(y2 + (h*scale -h)/2)

    if frame_shape is not None:
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame_shape[1])
        y2 = min(y2, frame_shape[0])

    return x1, y1, x2, y2

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

    
class Tracker:
    
    def __init__(self, frame, bb, tid):
        self.last_frame = frame
        self.t = dlib.correlation_tracker()
        self.t.start_track(frame, bb)
        self.tid = tid
        
    def intersect_area(self, bb):
        pos = self.t.get_position()
        tmp = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        return bb.intersect(tmp).area()
    
    def iou(self, bb):
        # intersection over union
        inter = self.intersect_area(bb)
        return inter / (inter + self.t.get_position().area() + bb.area())
    
    def update(self, frame):
        self.last_frame = frame
        return self.t.update(frame)
    
    def tracking_quality(self, frame, bb):
        # should be above 7
        ret = self.t.update(frame, bb)
        self.t.start_track(frame, bb)
        return ret
    
    def is_in_frame(self):
        # at least one pixel in the frame
        fheight, fwidth, _ = self.last_frame.shape
        pos = self.t.get_position()
        return (pos.right() > 0) and (pos.left() < fwidth) and (pos.top() < fheight) and (pos.bottom() > 0)
        


class TrackerList:
    
    def __init__(self):
        self.d = {}
        self.i = 0
        
    def update(self, frame):
        for fid in list(self.d):
            if self.d[fid].update(frame) < 7 or not self.d[fid].is_in_frame():
                del self.d[fid]

    def ingest_detection(self, frame, lbox):
        # le nombre de tracker restant doit être égal au nombre de bbox
        # au debut on envisage de tout enlever
        # pour toute bbox, si intersection, alors on enleve pas le match
        # si pas d'intersection, on enleve
        to_remove = list(self.d)
        to_add = {}
        
        for bb in lbox:
            rmscore = [self.d[k].tracking_quality(frame, bb) if self.d[k].iou(bb) > 0.5 else 0 for k in to_remove]
            am = np.argmax(rmscore) if len(rmscore) > 0 else None
            if am is not None and rmscore[am] > 7:
                # update tracker with the new bb coords
                idrm = to_remove[am]
                to_remove.remove(idrm)
                to_add[idrm] = Tracker(frame, bb, idrm)
            else:
                # new tracker
                to_add[self.i] = Tracker(frame, bb, self.i)
                self.i += 1
        self.d = to_add
  
    
  
    
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
    def __init__(self, threshold = 0.65, verbose = False):
        
        """ 
        The constructor for GenderVideo class. 
  
        Parameters: 
           threshold (float): quality of face detection considered acceptable, value between 0 and 1. 
        """
        
        p = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        self.face_detector = cv2.dnn.readNetFromTensorflow(p + "opencv_face_detector_uint8.pb",
                                                           p + "opencv_face_detector.pbtxt")
        self.align_predictor = dlib.shape_predictor(p +'shape_predictor_68_face_landmarks.dat')

        
        f = h5py.File(p + 'svm_classifier.hdf5', 'r')
        svm = LinearSVC()
        svm.classes_ = np.array(f['linearsvc/classes'][:]).astype('<U1')
        svm.intercept_ = f['linearsvc/intercept'][:]
        svm.coef_ = f['linearsvc/coef'][:]
        self.gender_svm = svm
        
        self.vgg_feature_extractor = VGGFace(include_top = False, input_shape = (224, 224, 3), pooling ='avg')
        self.threshold = threshold
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

    
    def align_and_crop_face(self, img, rect_list, desired_width, desired_height):
        """ 
        Aligns and resizes face to desired shape.
  
        Parameters: 
            img  : Image to be aligned and resized.
            rect_list: Bounding box coordinates tuples.
            desired_width: output image width.
            desired_height: output image height.
            
          
        Returns: 
            cropped_img: Image aligned and resized.
            left_eye: left eye position coordinates.
            right_eye: right eye position coordinates.
        """

        #### TODO Warning: this return a single element
        #### only the 1st element of rect_list is processed
        assert len(rect_list) == 1
        
        for j, det in enumerate(rect_list):
            shape = self.align_predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)
            M = get_rotation_matrix(left_eye, right_eye)

            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
            cropped = crop_image(rotated_img, det)
            cropped_res = cv2.resize(cropped, (desired_width, desired_height))
            cropped_img = cropped_res[:, :, ::-1]

            return cropped_img, left_eye, right_eye
        
    def detect_faces_from_image(self, img, bbox_scaling=1.1):
        """ 
        Detect faces from an image
  
        Parameters: 
            img (array): Image to detect faces from.
            desired_width (int): desired output width of the image.
            desired_height (int): desired output height of the image.
            bbox_scaling (float): scaling factor to the bounding box around the face.
          
        Returns: 
            faces_data (list) : List containing :
                                - the bounding box after scaling
                                - image cropped around the face and resized
                                - left eye coordinates
                                - right eye coordinates
                                - index of the face in the image 
                                - face detection confidence score
        """
        
        n_face = 0
        faces_data = []

        frame_height = img.shape[0]
        frame_width = img.shape[1]
        # The CNN is intended to work images resized to 300*300
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                n_face += 1
                bbox = _get_bbox_pts(detections, i, frame_width, frame_height)
                
                x1, y1, x2, y2 = bbox[:]
                x1, y1, x2, y2 = _scale_bbox(x1, y1, x2, y2, bbox_scaling, img.shape)
                
                
                if x1 < x2 and y1 < y2:
                    dets = [dlib.rectangle(x1, y1, x2, y2)]
                else:
                    ## TODO WARNING - THIS HACK IS STRANGE
                    dets = [dlib.rectangle(0, 0, frame_width, frame_height)]

                faces_data.append((dets, confidence))
        return faces_data



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
                faces_info = self.detect_faces_from_image(frame) 
                
                tl.ingest_detection(frame, [e[0][0] for e in faces_info])                
                
            # process faces based on position found in trackers
            for fid in tl.d:
                bb = tl.d[fid].t.get_position()
                
                x1, y1, x2, y2 = _scale_bbox(bb.left(), bb.top(), bb.right(), bb.bottom(), 1, frame.shape)
                if x1 < x2 and y1 < y2:
                    bb = [dlib.rectangle(x1, y1, x2, y2)]
                else:
                    ## TODO WARNING - THIS HACK IS STRANGE
                    bb = [dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])]
                
                
                face_img, left_eye, right_eye = self.align_and_crop_face(frame, bb, 224, 224)
                label, decision_value = self._gender_from_face(face_img)
                
                bb = bb[0]
                bb = '[(%d, %d) (%d, %d)]' % (bb.left(), bb.top(), bb.right(), bb.bottom())
                
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

            faces_info = self.detect_faces_from_image(frame)

            for bb, detect_conf in faces_info:
                face_img, left_eye, right_eye = self.align_and_crop_face(frame, bb, 224, 224)
                label, decision_value = self._gender_from_face(face_img)


                bb = bb[0]
                bb = '[(%d, %d) (%d, %d)]' % (bb.left(), bb.top(), bb.right(), bb.bottom()) 
#                print(bb, type(bb), str(bb))
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


