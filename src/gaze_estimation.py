import cv2
import numpy as np 
import logging as log

from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        '''
        This method instances variables for the Gaze Estimation Model.

        Args:
        model_name= Name of the face detection model
        device = Device used for inference (default = CPU)
        extensions = device extensions to add if there are unsupported layers
        threshold = confidence threshold for detection

        Return:
        None
        '''

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.model_supported=True
        self.device=device
        self.extensions=extensions
        self.threshold=threshold
        self._ie_core = IECore()

        try:
            try:
                self.model=self._ie_core.read_network(model=self.model_structure, weights=self.model_weights)
            except AttributeError:
                self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError('[Gaze Estimation Module] Could not initialize the network. Ensure that model path is correct')

        self.input_blob=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs['left_eye_image'].shape
        self.output_blob=next(iter(self.model.outputs))
        self.output_blob=self.model.outputs[self.output_blob].shape

        return None

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.

        Args:
        None

        Return:
        self.infer: Method for running inference on the input frames
        '''

        self.check_model()

        if self.model_supported == True:
            self.infer = self._ie_core.load_network(network=self.model, device_name=self.device, num_requests=1)
            log.info('[Gaze Estimation Module] Model loaded to IECore')

        return self.infer

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        This method runs inference on the input image.

        Args:
        left_eye_image = cropped left_eye image output of the facial landmarks detection model
        right_eye_image = cropped right_eye image output of the facial landmarks detection model
        head_pose_angles = 

        Return:
        gaze_x = x-coordinate of gaze. It is derived from the gaze vector
        gaze_y = y-coordinate of gaze. It is derived from the gaze vector
        '''
        
        self.head_pose_names='head_pose_angles'
        self.head_pose_angles = head_pose_angles
        self.left_eye_name='left_eye_image'
        self.left_eye_image = self.preprocess_input(left_eye_image)
        self.right_eye_name='right_eye_image'
        self.right_eye_image = self.preprocess_input(right_eye_image)        

        self.inference_handler=self.infer.start_async(request_id=0, inputs={self.left_eye_name: self.left_eye_image, 
                                                     self.right_eye_name: self.right_eye_image,
                                                     self.head_pose_names: self.head_pose_angles})

        if self.infer.requests[0].wait(-1)==0:
            gaze_vector = self.inference_handler.outputs
            gaze_x, gaze_y = self.preprocess_output(gaze_vector)

        return gaze_x, gaze_y

    def check_model(self):
        '''
        Checks if all models layers are supported by Inference engine and add the necessary extension
        if there are any unsupported layers

        Args:
        None

        Return:
        model_supported = A boolean value indicating if the model is supported by the Inference
                          engine (True) or not (False)
        '''

        supported_layers=self._ie_core.query_network(self.model, self.device)
        unsupported_layers=[layer for layer in self.model.layers.keys() if layer not in supported_layers]

        if (len(unsupported_layers) != 0) and (self.extension and self.device is not None):
            self.core.add_extension(self.extension, self.device)
            self.model_supported=True

            supported_layers = self._ie_core.query_network(self.model, self.device)
            unsupported_layers= [layer for layer in self.model.layers.keys() if layer not in supported_layers]
            if (len(unsupported_layers) != 0):
                log.error('[Inference Engine] Unsupported layers found in Gaze Estimation Model. Try using "CPU" as device argument (i.e. --device CPU)')
                log.error(f'[Inference Engine] Device used:{self.device}')
                self.model_supported=False

                if (self.device=="CPU"):
                    log.error('[Inference Engine] CPU was used but some layers are not supported, Create custom layers for unsupported layers')
        
                exit(1)
        
            else:
                self.model_supported = True

        return self.model_supported

    def preprocess_input(self, image):
        '''
        Preprocesses the data before feeding it into the network for inference.

        Args
        Image = frsme or image that inference will be performed upon

        Return:
        p_image = preprocessed frame or image
        '''

        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)

        return p_image

    def preprocess_output(self, gaze_vector):
        '''
        Extracts the x and y coordinates of the face's gaze from the gaze vector output of the inference

        Args:
        gaze_vector = a [1,3] blob of 2D coordinates of the gaze of the left and right eyes

        Return:
        gaze_x = x coordinate of the gaze
        gaze_y = y coordinate of the gaze
        '''
        
        gaze_x = gaze_vector.get('gaze_vector')[0][0]
        gaze_y = gaze_vector.get('gaze_vector')[0][1]
        log.info('[ Gaze Estimator ] Acquired gaze vector coordinates')
        
        return gaze_x, gaze_y