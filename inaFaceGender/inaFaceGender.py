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
import csv
from .face_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from sklearn.externals import joblib
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image 


def write_to_video(frames_list, file_name, fps):
    """ 
    Writes a list of frames into a video using MP4V encoding. 
  
    Parameters: 
    frames_list (list): List of the frames to write
    file_name (string): video output path
    fps (int) : Number of frames per second used in output video
  
  
    """
    frame_width = frames_list[0].shape[0]
    frame_height = frames_list[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(file_name,fourcc,
                      fps, (frame_height,frame_width))

    for frame in frames_list:
        
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out.release()

    
def _get_bbox_pts(detections, face_idx, frame_width, frame_height):
    
    
    x1 = int(detections[0, 0, face_idx, 3] * frame_width)
    y1 = int(detections[0, 0, face_idx, 4] * frame_height)
    x2 = int(detections[0, 0, face_idx, 5] * frame_width)
    y2 = int(detections[0, 0, face_idx, 6] * frame_height)

    width = x2 - x1
    height = y2 - y1
    max_size = max(width, height)
    x1, x2 = max(0, (x1 + x2) // 2 - max_size // 2), min(frame_width, (x1 + x2) // 2 + max_size // 2)
    y1, y2 = max(0, (y1 + y2) // 2 - max_size // 2), min(frame_height, (y1 + y2) // 2 + max_size // 2)

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


def _match_bbox_tracker(bbox, tracker):
    # bbox info
    x = bbox.left()
    y = bbox.top()
    width = bbox.width()
    height = bbox.height()

    x_center = x + 0.5 * width
    y_center = y + 0.5 * height

    # tracker info
    tracked_position =  tracker.get_position()

    t_x = int(tracked_position.left())
    t_y = int(tracked_position.top())
    t_w = int(tracked_position.width())
    t_h = int(tracked_position.height())
    
    t_x_center = t_x + 0.5 * t_w
    t_y_center = t_y + 0.5 * t_h
    
    return ( ( t_x <= x_center   <= (t_x + t_w)) and 
         ( t_y <= y_center   <= (t_y + t_h)) and 
         ( x  <= t_x_center <= (x + width)) and 
         ( y   <= t_y_center <= (y + height)))

    
def is_tracker_pos_in_frame(tracker, frame):
    fheight, fwidth, _ = frame.shape
    pos = tracker.get_position()
    #print('tracker pos in frame', pos.right(), pos.left(), pos.top(), pos.bottom())
    return (pos.right() > 0) and (pos.left() < fwidth) and (pos.top() < fheight) and (pos.bottom() > 0)

    
    
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
        self.gender_svm = joblib.load(p + 'svm_classifier.joblib')
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

        
    def _process_tracked_face(self, cur_tracker, frame):
        
        ## There is no rotation in this function... results may be suspicious
        
        
        tracked_position =  cur_tracker.get_position()
        #print('tracked position', tracked_position)
        #print('frame_shape', frame.shape)
#        print('cur_tracker', cur_tracker)

        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())
        
#        print('tracked face: id, x, y, w, h', face_id, t_x, t_y, t_w, t_h)

        copy_img = frame[max(0, t_y):(t_y + t_h), max(0, t_x):(t_x + t_w)]

        
        #print('simage shape', copy_img.shape)
        copy_img = cv2.resize(copy_img, (224,224))

        label, decision_value = self._gender_from_face(copy_img)
        
        return (t_x, t_y, t_w, t_h, label, decision_value)

    
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
        
        for j, det in enumerate(rect_list):
            shape = self.align_predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)
            M = get_rotation_matrix(left_eye, right_eye)

            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
            cropped = crop_image(rotated_img, det)
            try:
                
                cropped_res = cv2.resize(cropped, (desired_width, desired_height))
            except:
                print('except in align_and_crop_faces', det)
                print(img.shape)
                cropped_res = cv2.resize(rotated_img,(desired_width, desired_height))
            cropped_img = cropped_res[:, :, ::-1]

            return cropped_img, left_eye, right_eye
        
    def detect_faces_from_image(self, img, desired_width,
                                        desired_height, bbox_scaling=1.1):
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
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                n_face += 1
                bbox = _get_bbox_pts(detections, i, frame_width, frame_height)
                x1, y1 = [int(i * abs(bbox_scaling//1 - bbox_scaling%1)) for i in bbox[:2]]
                x2, y2 = [int(i*bbox_scaling) for i in bbox[2:]]
                if x1 < x2 and y1 < y2:
                    dets = [dlib.rectangle(x1, y1, x2, y2)]
                else:
                    dets = [dlib.rectangle(0, 0, frame_width, frame_height)]

                  
                face_img, left_eye, right_eye = self.align_and_crop_face(img, dets, desired_width,
                                                                    desired_height)
                
                face_data = [dets, face_img, left_eye, right_eye,
                             'face_%d' % n_face, confidence]
                faces_data.append(face_data)

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

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        current_face_id = 0
        
        face_trackers = {}
        confidence = {}

        info = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Video file does not exist or is invalid")
        
        while cap.isOpened() :
            ret, frame = cap.read()
            if not ret:
                break

            # skip frames until offset is reached or for subsampling reasons
            if (cap.get(cv2.CAP_PROP_POS_MSEC) < offset) or (cap.get(cv2.CAP_PROP_POS_FRAMES) % subsamp_coeff != 0):
                continue

            #if ((cap.get(cv2.CAP_PROP_POS_FRAMES)) % 1000 == 0) or True:
            #    print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            #    print('dface trackers before update', face_trackers)
                
                
            # track faces in current frame
            face_ids_to_delete = []
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for fid in face_trackers:
                tracking_quality = face_trackers[fid].update(frame)

                if (tracking_quality < 7) or (not is_tracker_pos_in_frame(face_trackers[fid], frame)):
                    face_ids_to_delete.append(fid)

            for fid in face_ids_to_delete:
                face_trackers.pop(fid)
            #print('dface trackers after update', face_trackers)

            # detect faces every k frames
            if (cap.get(cv2.CAP_PROP_POS_FRAMES) % k_frames)==0:
                faces_info = self.detect_faces_from_image(frame,
                                      desired_width=224,  desired_height=224) 
                if faces_info:
                    for element in faces_info:
                        bbox = element[0][0]
                        confidence[ current_face_id ] = round(element[5], 3)

                        matched_fid = None

                        # match detected face to previously tracked faces
                        for fid in face_trackers:
                            ## TODO/BUG: several elements may match using this condition
                            ## This loop should be debugged to use the closest match found,
                            ## instead of the last match found
                            if _match_bbox_tracker(bbox, face_trackers[fid]):
                                matched_fid = fid

                        # if detected face is not corresponding to previously tracked faces
                        # create a new face id and a new face tracker
                        # BUG: in the current implementation, the newly detected face bounding box
                        # is not used to update the tracker bounding box
                        if matched_fid is None:

                            tracker = dlib.correlation_tracker()
                            tracker.start_track(frame, bbox)

                            face_trackers[ current_face_id ] = tracker
                            current_face_id += 1
                #print('dface trackers after face detection ', face_trackers)

            # delete invalide face positions
            face_ids_to_delete = []
            for fid in face_trackers:
                if not is_tracker_pos_in_frame(face_trackers[fid], frame):
                    face_ids_to_delete.append(fid)
            for fid in face_ids_to_delete:
                face_trackers.pop(fid)                
                
                
                
            # process faces based on position found in trackers
            for fid in face_trackers:
                t_x, t_y, t_w, t_h, label, decision_value = self._process_tracked_face(face_trackers[fid], frame)
                t_bbox = dlib.rectangle(t_x, t_y, t_x+t_w, t_y+t_h)
                info.append([
                        cap.get(cv2.CAP_PROP_POS_FRAMES), fid,  t_bbox, (t_w, t_h), label,
                        decision_value, confidence[fid]
                    ])



        cap.release()
        track_res = pd.DataFrame.from_records(info, columns = ['frame', 'faceid', 'bb', 'size','label', 'decision', 'conf'])
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
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Video file does not exist or is invalid")
        
        info = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # skip frames until offset is reached or for subsampling reasons
            if (cap.get(cv2.CAP_PROP_POS_MSEC) < offset) or (cap.get(cv2.CAP_PROP_POS_FRAMES) % subsamp_coeff != 0):
                continue


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_info = self.detect_faces_from_image(frame,
                                                      desired_width=224,  desired_height=224)      
            if faces_info:
                for element in faces_info:
                    label, decision_value = self._gender_from_face(element[1])
                    bounding_box = element[0][0]
                    detection_score = round(element[5], 3)
                    bbox_length = bounding_box.bottom() - bounding_box.top()

                    info.append([
                        cap.get(cv2.CAP_PROP_POS_FRAMES), bounding_box, (bbox_length, bbox_length), label,
                        decision_value, detection_score
                    ])

        cap.release()
        info = pd.DataFrame.from_records(info, columns = ['frame', 'bb', 'size','label', 'decision', 'conf'])
        return info
    


