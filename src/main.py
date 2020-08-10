'''
To use the demo video file with default inputs:
python3 main.py --input_type 'demo' --precision medium

To use the device camera with default inputs:
python3 main.py --input_type cam
'''

import os
import cv2
import sys
import time

import numpy as np
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder 
from face_detection import FaceDetection 
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController 
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection 


INPUT=['video','cam','image','demo']
DEVICES=['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO:GPU,CPU', 'MULTI:MYRIAD,GPU,CPU',
         'MULTI:MYRIAD,CPU', 'HETERO:FPGA,MYRIAD,GPU', 'HETERO:FPGA,CPU', 'HETERO:FPGA,GPU']
SPEED=['fast', 'medium', 'slow']
PRECISION=['high', 'medium', 'low']
MODEL_PRECISION=['FP32','FP16','INT8']
BATCH = range(1,200)


def build_argparser():
    '''
    Parses commandline inputs

    Args:
    None

    Return:
    All parsed command line arguments
    '''

    parse=ArgumentParser()

    parse.add_argument('-i', '--input_type', default='cam', choices=INPUT, type=str, metavar='INPUT', required=True, 
                        help='(required) specify the input type (Default is "CAM")'\
                             '\nchoices: [%(choices)s]')
    parse.add_argument('-if', '--input_file', default=None, type=str, metavar='INPUT_PATH',
                        required=False, help='(optional) Path to input video')
    parse.add_argument('-s', '--speed', default='fast', choices=SPEED, type=str, metavar='SPEED', required=False, 
                        help='(optional) Specify mouse pointer movement speed (Default is "medium")'\
                             '\nchoices: [%(choices)s]')
    parse.add_argument('-p', '--precision', default='medium', choices=PRECISION, type=str, metavar='PRECISION', required=False, 
                        help='(optional) Specify mouse pointer movement precision (Default is "high")'\
                             '\nchoices: [%(choices)s]')

    parse.add_argument('-fm','--face_model', default='intel/face-detection-adas-0001/FP16/face-detection-adas-0001',
                        type=str, metavar='MODEL_PATH', required=False, 
                        help='(optional) Path to model xml file (enter the file name without .xml extension)'\
                                'Precisions available: [FP32, FP16, INT8]')
    parse.add_argument('-fde', '--face_device_ext', default=None, type=str, metavar='DEVICE_EXTENSION',
                        required=False, help='(optional) Path to device extension library with the kernels implementation.')
    parse.add_argument('-fd', '--face_device', default='CPU', choices=DEVICES, type=str, metavar='DEVICE', required=False, 
                        help='(optional) Specify target device for inference. (Default is "CPU")'\
                             '\nchoices: [%(choices)s]')
    parse.add_argument('-fpt', '--face_prob_threshold', default=0.5, type=float, metavar='PROB_THRESHOLD', 
                        required=False, help='(optional) Probability threshold for Face detection model. (Default is 0.5)')

    parse.add_argument('-flm','--face_landmark_model', default='intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009', 
                        type=str, metavar='MODEL_PATH', required=False, 
                        help='(optional) Path to model xml file (enter the file name without .xml extension)'\
                             'Precisions available: [FP32, FP16, INT8]')
    parse.add_argument('-flde', '--face_landmark_device_ext', default=None, type=str,metavar='DEVICE_EXTENSION',
                        required=False, help='(optional) Path to device extension library with the kernels implementation.')
    parse.add_argument('-fld', '--face_landmark_device', default='CPU', choices=DEVICES, type=str, metavar='DEVICE', required=False, 
                        help='(optional) Specify target device for inference. (Default is "CPU")'\
                             '\nchoices: [%(choices)s]')
    parse.add_argument('-flpt', '--face_landmark_prob_threshold', default=0.5, type=float, metavar='PROB_THRESHOLD', 
                        required=False, help='(optional) Probability threshold for the Facial landmark detection model. (Default is 0.5)')

    parse.add_argument('-hpm','--head_pose_model', default='intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001',
                        type=str, metavar='MODEL_PATH', required=False, 
                        help='(optional) Path to model xml file (enter the file name without .xml extension)'\
                             'Precisions available: [FP32, FP16, INT8]')
    parse.add_argument('-hpde', '--head_pose_device_ext', default=None, type=str, metavar='DEVICE_EXTENSION',
                        required=False, help='(optional) Path to device extension library with the kernels implementation.')
    parse.add_argument('-hpd', '--head_pose_device', default='CPU', choices=DEVICES, type=str, metavar='DEVICE', required=False, 
                        help='(optional) Specify target device for inference. (Default is "CPU")'\
                             '\nchoices: [%(choices)s]')
    parse.add_argument('-hppt', '--head_pose_prob_threshold', default=0.5, type=float, metavar='PROB_THRESHOLD', 
                        required=False, help='(optional) Probability threshold for Head Pose detection model. (Default is 0.5)\n')

    parse.add_argument('-gm', '--gaze_model', default='intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002',
                        type=str, metavar='MODEL_PATH', required=False, 
                        help='(optional) Path to model xml file (enter the file name without .xml extension)'\
                             'Precisions available: [FP32, FP16, INT8]')
    parse.add_argument('-gde', '--gaze_device_ext', default=None, type=str, metavar='DEVICE_EXTENSION',
                        required=False, help='(optional) Path to device extension library with the kernels implementation.')
    parse.add_argument('-gd', '--gaze_device', default='CPU', choices=DEVICES, type=str, metavar='DEVICE', required=False, 
                        help='(optional) Specify target device for inference. (Default is "CPU") \nchoices: [%(choices)s]')
    parse.add_argument('-gpt', '--gaze_prob_threshold', default=0.5, type=float, metavar='PROB_THRESHOLD', 
                        required=False, help='(optional) Probability threshold for Face detection model. (Default is 0.5).')

    parse.add_argument('-all_dev', '--all_devices', default='CPU', choices=DEVICES, type=str, metavar='DEVICE', required=False, 
                        help='(optional) Specify all target devices at once. (Default is "CPU") \nchoices: [%(choices)s]')
    parse.add_argument('-b', '--batch_size', default=30, choices=BATCH, type=int, metavar='BATCH',
                        required=False, help='(optional) Batch size of input for inference. (Default is 30)')
    parse.add_argument('--output_path', default='outputs/', type=str, metavar='OUTPUT_PATH',
                        required=False, help='(optional) Specify output directory (Default is outputs/).')
    parse.add_argument('--show_output', default=False, type=bool, metavar='SHOW_OUTPUTS',
                        required=False, help='(optional) Display detection results and draw face and eyes bounding'\
                                             'boxes on the video frame (Default is False).')
            
    return parse


class MoveMouse:
    '''
    Main Class for the Mouse Controller app. 
    This is the class where all the models are stitched together to control the mouse pointer
    '''
    def __init__(self, args):
        '''
        This method instances variables for the Facial Landmarks Detection Model.

        Args:
        args = All arguments parsed by the arguments parser function

        Return:
        None
        '''

        init_start_time=time.time()
        self.output_path=args.output_path
        self.show_output=args.show_output
        self.total_processing_time=0
        self.count_batch=0
        self.inference_speed = []
        self.avg_inference_speed = 0

        if args.all_devices!='CPU':
            args.face_device = args.all_devices
            args.face_landmark_device = args.all_devices
            args.head_pose_device = args.all_devices
            args.gaze_device = args.all_devices

        model_init_start = time.time()
        self.face_model=FaceDetection(args.face_model, args.face_device, 
                                     args.face_device_ext, args.face_prob_threshold)
        self.landmarks_model=FacialLandmarksDetection(args.face_landmark_model, 
                                                     args.face_landmark_device, 
                                                     args.face_landmark_device_ext, 
                                                     args.face_landmark_prob_threshold)
        self.head_pose_model=HeadPoseEstimation(args.head_pose_model, 
                                                args.head_pose_device,
                                                args.head_pose_device_ext, 
                                                args.head_pose_prob_threshold)
        self.gaze_model=GazeEstimation(args.gaze_model, args.gaze_device, 
                                       args.gaze_device_ext, args.gaze_prob_threshold)
        self.model_init_time = time.time() - model_init_start
        log.info('[ Main ] All required models initiallized')

        self.mouse_control=MouseController(args.precision, args.speed)
        log.info('[ Main ] Mouse controller successfully initialized')

        self.input_feeder=InputFeeder(args.batch_size, args.input_type, args.input_file)
        log.info('[ Main ] Initialized input feeder')

        model_load_start=time.time()
        self.face_model.load_model()
        self.landmarks_model.load_model()
        self.head_pose_model.load_model()
        self.gaze_model.load_model()

        self.model_load_time=time.time() - model_load_start
        self.app_init_time=time.time() - init_start_time
        log.info('[ Main ] All moadels loaded to Inference Engine\n')

        return None
    
    def draw_face_box(self, frame, face_coords):
        '''
        Draws face's bounding box on the input frame
        Args:
        frame = Input frame from video or camera feed. It could also be an input image

        Return:
        frame = Frame with bounding box of faces drawn on it
        '''

        start_point=(face_coords[0][0], face_coords[0][1])
        end_point=(face_coords[0][2], face_coords[0][3])
        thickness=5
        color=(255, 86, 0)

        frame=cv2.rectangle(frame, start_point, end_point, color, thickness)

        return frame

    
    def draw_eyes_boxes(self, frame, left_eye_coords, right_eye_coords):
        '''
        Draws face's bounding box on the input frame
        Args:
        frame = Input frame from video or camera feed. It could also be an input image

        Return:
        frame = Frame with bounding box of left and right eyes drawn on it
        '''

        left_eye_start_point=(left_eye_coords[0], left_eye_coords[1])
        left_eye_end_point=(left_eye_coords[2], left_eye_coords[3])
        right_eye_start_point=(right_eye_coords[0], right_eye_coords[1])
        right_eye_end_point=(right_eye_coords[2], right_eye_coords[3])
        thickness=5
        color=(0, 210, 0)

        frame=cv2.rectangle(frame, left_eye_start_point, left_eye_end_point, color, thickness)
        frame=cv2.rectangle(frame, right_eye_start_point, right_eye_end_point, color, thickness)

        return frame


    def draw_outputs(self, frame):
        '''
        Draws the inference outputs (bounding boxes of the face and both eyes and 
        the 3D head pose directions) of the four models onto the frames.

        Args:
        frame = Input frame from video or camera feed. It could also be an input image

        Return:
        frame = Frame with all inference outputs drawn on it
        '''

        frame=self.draw_face_box(frame, self.face_coords)
        frame=self.draw_eyes_boxes(frame, self.left_eye_coords, self.right_eye_coords)

        frame_id = f'Batch id = {self.count_batch}'
        avg_inference_speed=f'Avg. inference speed = {self.avg_inference_speed:.3f}fps'
        total_processing_time=f'Total infer. time = {self.total_processing_time:.3f}s'

        cv2.putText(frame, frame_id, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 86, 0), 1)
        cv2.putText(frame, avg_inference_speed, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 86, 0), 1)
        cv2.putText(frame, total_processing_time, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 86, 0), 1)
        
        return frame
        


    def run_inference(self, frame):
        '''
        Performs inference on the input video or image by passing it through all four
        models to get the desired coordinates for moving the mouse pointer.

        Args:
        frame = Input image, frame from video or camera feed

        Return:
        None
        '''

        self.input_feeder.load_data()

        for frame in self.input_feeder.next_batch():

            if self.input_feeder.frame_flag==True:
                log.info('[ Main ] Started processing a new batch')
                start_inference = time.time()
                self.face_coords, self.face_crop=self.face_model.predict(frame)
                
                if self.face_coords == []:
                    log.info('[ Main ] No face detected.. Waiting for you to stare at the camera')
                    f.write('[ Error ] No face was detected')

                else:
                    self.head_pose_angles=self.head_pose_model.predict(self.face_crop)
                    self.left_eye_coords, self.left_eye_image, self.right_eye_coords, self.right_eye_image=self.landmarks_model.predict(self.face_crop)
                    self. x, self.y = self.gaze_model.predict(self.left_eye_image, self.right_eye_image, self.head_pose_angles)
                    log.info(f'[ Main ] Relative pointer coordinates: [{self.x:.2f}, {self.y:.2f}]')

                    batch_process_time = time.time() - start_inference
                    self.total_processing_time += batch_process_time
                    self.count_batch += 1
                    log.info(f'[ Main ] Finished processing batch. Time taken = {batch_process_time}s\n')

                    self.mouse_control.move(self.x, self.y)

                    if self.show_output:
                        self.draw_outputs(frame)

                    cv2.imshow('Computer Pointer Controller Output', frame)
                    self.inference_speed.append( self.count_batch/ self.total_processing_time)
                    self.avg_inference_speed = sum(self.inference_speed)/len(self.inference_speed)

                with open(os.path.join(self.output_path, 'outputs.txt'), 'w+') as f:
                    f.write('INFERENCE STATS\n')
                    f.write(f'Total model initialization time : {self.model_init_time:.2f}s\n')
                    f.write(f'Total model load time: {self.model_load_time:.2f}s\n')
                    f.write(f'App initialization time: {self.app_init_time:.2f}s\n')
                    f.write(f'Total processing time: {self.total_processing_time:.2f}s\n')
                    f.write(f'Average inference speed: {self.avg_inference_speed:.2f}FPS\n')
                    f.write(f'Batch count: {self.count_batch}\n\n')
                    
                    f.write('LAST OUTPUTS\n')
                    f.write(f'Face coordinates: {self.face_coords}\n')
                    f.write(f'Left eye coordinates: {self.left_eye_coords}\n')
                    f.write(f'Right eye coordinates: {self.right_eye_coords}\n')
                    f.write(f'Head pose angles: {self.head_pose_angles}\n')
                    f.write(f'Relative pointer coordinates/ Gaze vector: [{self.x:.2f}, {self.y:.2f}]')
            
            else:
                self.input_feeder.close()
                cv2.destroyAllWindows()

                log.info(f'[ Main ] All input Batches processed in {self.total_processing_time:.2f}s')
                log.info('[ Main ] Shutting down app...')
                log.info('[ Main ] Mouse controller app has been shut down.')
                break

        return 


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format='[%(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.debug(str(args))

    controller = MoveMouse(args)
    controller.run_inference(args)

if __name__ == '__main__':
    main()