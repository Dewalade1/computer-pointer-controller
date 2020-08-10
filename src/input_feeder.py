'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''

import os
import cv2
import math
import logging as log

from numpy import ndarray

IMAGE_FORMATS=['.jpg', '.bmp', '.dpx', '.png', '.gif', '.webp', '.tiff', '.psd', '.raw',
                '.heif', '.indd']
VIDEO_FORMATS=['.mp4', '.webm', '.mpg', '.mp2', '.mpeg', '.mpe', '.mpv', '.ogg', '.m4p',
                '.m4v', '.avi', '.wmv', '.mov', '.qt', '.flv', '.swf', '.avchd']

class InputFeeder:
    def __init__(self, batch_size, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        batch_size: int, The batch_size for te input batch
        '''
        
        self.input_type=input_type.lower()
        self.batch_size=batch_size

        if self.input_type=='video' or self.input_type=='image':
            self.input_file=input_file
        elif self.input_type=='demo':
            self.input_file='bin/demo.mp4'

        if self.input_type!='cam':
            self.input_extension=os.path.splitext(self.input_file)[1].lower()
    
    def load_data(self):

        if (self.input_type=='video' or self.input_type=='demo') and (self.input_extension in VIDEO_FORMATS):
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        elif (self.input_type=='image') and (self.input_extension in IMAGE_FORMATS):
            self.cap=cv2.imread(self.input_file)
        else:
            log.error('[ Input feeder] Input file not supported. Please use the proper video or image format')

    def next_batch(self):
        '''
        Collects and returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.

        Args:
        None

        Returns:
        frame = next frame in video file
        '''
        
        while True:
            for _ in range(self.batch_size):
                _, frame=self.cap.read()
                self.frame_flag=_
            
                if not _:
                    break

            yield frame


    def close(self):
        '''
        Closes the VideoCapture.

        Args:
        None

        Returns:
        None
        '''
        if self.input_type != 'image':
            self.cap.release()

