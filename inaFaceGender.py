import dlib, cv2
import numpy as np
import pandas as pd
import os
import csv
from face_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
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
    def __init__(self, threshold = 0.65):
        
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
        
    def _process_tracked_face(self, face_id, trackers, frame):
        
        tracked_position =  trackers[face_id].get_position()

        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        if t_y < 0 and t_x < 0:
            copy_img = frame[0:(t_y+t_h), 0:(t_x+t_w)]
        elif t_y < 0:
            copy_img = frame[0:(t_y+t_h), t_x:(t_x+t_w)]    
        elif t_x < 0:
            copy_img = frame[t_y:(t_y+t_h), 0:(t_x+t_w)]     
        else:
            copy_img = frame[t_y:(t_y+t_h), t_x:(t_x+t_w)]

        copy_img = cv2.resize(copy_img, (224,224))    
        face_img = image.img_to_array(copy_img)

        face_img = utils.preprocess_input(face_img, version=1)
        face_img = np.expand_dims(face_img, axis=0)
      
        features = self.vgg_feature_extractor.predict(face_img)
        
        
       
        label = self.gender_svm.predict(features)[0]
        
        decision_value = round(self.gender_svm.decision_function(features)[0], 3)
        
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
                print(det)
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
    
    
    def detect_with_tracking(self, video_path, k_frames, per_frames = 1, offset = None):
        """
        Pipeline for gender classification from videos using correlation filters based tracking (dlib's).
  
        Parameters: 
            video_path (string): Path for input video.
            k_frames (int) : Number of frames for which continue tracking the faces without renewing face detection.
            per_frames (int) : Number of frames to skip processing.
            offset (float) : Time in milliseconds to skip at the beginning of the video.
          
        Returns: 
            info (DataFrame): A Dataframe with frame and face information (coordinates, decision function, smoothed and non smoothed labels)
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        current_face_id = 0
        current_frame = 0
        
        face_trackers = {}
        confidence = {}

        info = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Video file does not exist or is invalid")

        if offset:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset)
        
        while cap.isOpened() :
            ret, frame = cap.read()
            if ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % per_frames == 0:
                    face_ids_to_delete = []
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for fid in face_trackers.keys():
                        tracking_quality = face_trackers[ fid ].update( frame )

                        if tracking_quality < 7:
                            face_ids_to_delete.append( fid )

                    for fid in face_ids_to_delete:
                        face_trackers.pop(fid)

                    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % k_frames)==0 or cap.get(cv2.CAP_PROP_POS_FRAMES)  == 1:
                        faces_info = self.detect_faces_from_image(frame,
                                              desired_width=224,  desired_height=224) 
                        if faces_info:
                            for element in faces_info:
                                bbox = element[0][0]
                                confidence[ current_face_id ] = round(element[5], 3)
                                x = bbox.left()
                                y = bbox.top()
                                width = bbox.width()
                                height = bbox.height()

                                x_center = x + 0.5 * width
                                y_center = y + 0.5 * height

                                matched_fid = None

                                for fid in face_trackers.keys():
                                    tracked_position =  face_trackers[fid].get_position()

                                    t_x = int(tracked_position.left())
                                    t_y = int(tracked_position.top())
                                    t_w = int(tracked_position.width())
                                    t_h = int(tracked_position.height())

                                    t_x_center = t_x + 0.5 * t_w
                                    t_y_center = t_y + 0.5 * t_h

                                    if ( ( t_x <= x_center   <= (t_x + t_w)) and 
                                         ( t_y <= y_center   <= (t_y + t_h)) and 
                                         ( x  <= t_x_center <= (x + width)) and 
                                         ( y   <= t_y_center <= (y + height))):
                                        matched_fid = fid

                                if matched_fid is None:

                                    tracker = dlib.correlation_tracker()
                                    tracker.start_track(frame,
                                                        dlib.rectangle( x,
                                                                        y,
                                                                        x+width,
                                                                        y+height))

                                    face_trackers[ current_face_id ] = tracker
                                    current_face_id += 1

                    for fid in face_trackers.keys():
                        t_x, t_y, t_w, t_h, label, decision_value = self._process_tracked_face(fid, face_trackers, frame)
                        t_bbox = dlib.rectangle(t_x, t_y, t_x+t_w, t_y+t_h)
                        info.append([
                                cap.get(cv2.CAP_PROP_POS_FRAMES), fid,  t_bbox, (t_w, t_h), label,
                                decision_value, confidence[fid]
                            ])


            else: 
                break
        cap.release()
        track_res = pd.DataFrame.from_records(info, columns = ['frame', 'faceid', 'bb', 'size','label', 'decision', 'conf'])
        info = _smooth_labels(track_res)
        
        return info


    def __call__(self, video_path, per_frames = 1 ,  offset = None):
        
        """
        Pipeline function for gender classification from videos without tracking.
  
        Parameters: 
            video_path (string): Path for input video. 
            per_frames (int) : Number of frames to skip processing.
            offset (float) : Time in milliseconds to skip at the beginning of the video.
          
          
        Returns: 
            info: A Dataframe with frame and face information (coordinates, decision function,labels..)
        """
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Video file does not exist or is invalid")

        
        if offset:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset)
        
        
        info = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % per_frames == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces_info = self.detect_faces_from_image(frame,
                                                              desired_width=224,  desired_height=224)      
                    if faces_info:
                        for element in faces_info:
                            face_img = image.img_to_array(element[1])

                            face_img = utils.preprocess_input(face_img, version=1)
                            face_img = np.expand_dims(face_img, axis=0)

                            features = self.vgg_feature_extractor.predict(face_img)
                            label = self.gender_svm.predict(features)[0]
                            decision_value = round(self.gender_svm.decision_function(features)[0], 3)

                            bounding_box = element[0][0]
                            detection_score = round(element[5], 3)
                            bbox_length = bounding_box.bottom() - bounding_box.top()

                            info.append([
                                cap.get(cv2.CAP_PROP_POS_FRAMES), bounding_box, (bbox_length, bbox_length), label,
                                decision_value, detection_score
                            ])

            else:
                break
        cap.release()
        info = pd.DataFrame.from_records(info, columns = ['frame', 'bb', 'size','label', 'decision', 'conf'])
        return info
    


