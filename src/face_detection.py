import cv2
import numpy as np
import logging as log

from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        '''
        This method instances the variables for the Face Detection Class.

        Args:
        model_name= Name of the face detection model
        device = Device used for inference (default = CPU)
        Extensions = device extensions to add if there are unsupported layers
        Threshold = confidence threshold for detection

        Return:
        None
        '''
        
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.model_supported = True
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
            raise ValueError('[Face Detection Module] Could not initialize the network. Ensure that model path is correct')

        # self.input_name=next(iter(self.model.inputs))
        self.input_name=next(iter(self.model.input_info))
        # self.input_shape=self.model.input_info[self.input_name].shape 
        self.input_shape = self.model.input_info[self.input_name].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        return None


    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.

        Args:
        None

        Return:
        self.infer = Method for running inference on the input frames
        '''

        self.check_model()

        if self.model_supported == True:
            self.infer = self._ie_core.load_network(network=self.model, device_name=self.device, num_requests=1)
            log.info(msg = '[Face Detection Module] Model loaded to IECore')

        return self.infer

    def predict(self, image):
        '''
        This method runs inference on the input image.

        Args:
        image = Single or batch of input images or frames that have gone through inference

        Return:
        face_output = Cropped face that is created using the preprocess_output method
        crop_coords =  list of coordinates of all the bounding box edges of all croped 
                       faces.
        '''
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

        self.input_image = self.preprocess_input(image)
        # self.infer.start_async(request_id=0, inputs={self.input_name: self.input_image})
        self.infer_request = self.infer.start_async(request_id=0, inputs={self.input_name: self.input_image})

        # if self.infer.requests[0].wait(-1)==0:
        #     get_output = self.infer.requests[0].outputs[self.output_name]
        #     crop_coords, face_output = self.preprocess_output(image, get_output)

        if self.infer_request.wait() == 0:
            get_output = self.infer_request.output_blobs[self.output_name]
            crop_coords, face_output = self.preprocess_output(image, get_output)

        return crop_coords, face_output

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
        
        supported_layers = self._ie_core.query_network(self.model, self.device)
        # print(supported_layers)
        # unsupported_layers = [layer for layer in self.model.layers.keys() if layer not in supported_layers]
        unsupported_input_layers = [layer for layer in self.model.input_info.keys() if layer not in supported_layers]
        # print(unsupported_input_layers)
        unsupported_output_layers = [layer for layer in self.model.outputs.keys() if layer not in supported_layers]
        # print(unsupported_output_layers)
        unsupported_layers = unsupported_input_layers + unsupported_output_layers
        
        if (len(unsupported_layers) != 0) and (self.extension) and (self.device is not None):
            self.core.add_extension(self.extension, self.device)
            self.model_supported = True

            supported_layers = self._ie_core.query_network(self.model, self.device)
            unsupported_layers= [layer for layer in self.model.layers.keys() if layer not in supported_layers]
            if (len(unsupported_layers) != 0):
                log.error('[Inference Engine] Unsupported layers found in Face Detection Model. Try using "CPU" as device argument (i.e. --device CPU)')
                log.error(f'[Inference Engine] Device used:{self.device}')
                self.model_supported = False

                if (self.device=="CPU"):
                    log.error('[Inference Engine] CPU was used but some layers are not supported, Create custom layers for unsupported layers')

                exit(1)

        return self.model_supported

    def preprocess_input(self, image):
        '''
        Preprocesses the data before feeding it into the model for inference.

        Args
        Image = frame or image that inference will be performed upon

        Return:
        p_image = preprocessed frame or image
        '''
        
        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)

        return p_image

    def preprocess_output(self, image, outputs):
        '''
        Used bounding coordates extracted from the inference outputs to crop 
        all detected faces from a frame or an image. 

        Args:
        image = Single or batch of input images or frames that have gone through inference
        outputs = Results of inference on input frames or images

        Return:
        cropped_face = batches of faces cropped out of input image
        crop_coords =  list of coordinates of all the bounding box edges of all croped 
                       faces.
        '''
        
        crop_coords = []
        for bounding_box in outputs.buffer[0][0]:
            if bounding_box[2] >= self.threshold:
                xmin = int(bounding_box[3]*self.image_width)
                ymin = int(bounding_box[4]*self.image_height)
                xmax = int(bounding_box[5]*self.image_width)
                ymax = int(bounding_box[6]*self.image_height)

                crop_coords.append([xmin,ymin,xmax,ymax])
                cropped_face = image[ymin: ymax, xmin: xmax]
        log.info('[ Face Detector ] Cropped faces extracted')
        
        return crop_coords, cropped_face
