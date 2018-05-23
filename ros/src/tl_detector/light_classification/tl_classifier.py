from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
import os
#import six.moves.urllib as urllib
#import tarfile
import tensorflow as tf
from os import path


class TLClassifier(object):
    def __init__(self,RCNN_CLASSIFIER):
	
        #TODO load classifier
	rospy.logwarn("Initiating TL Classifier object ..")
	self.image = None
	
	self.lower_red_a = np.array([0,50,50])
	self.upper_red_a = np.array([10,255,255])
	self.lower_red_b = np.array([170,50,50])
	self.upper_red_b = np.array([180,255,255])
	
	if(RCNN_CLASSIFIER is True):
		self.working_dir = os.path.dirname(os.path.realpath(__file__))
		self.gpath = self.working_dir + '/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph_t_1_4.pb'

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
	

	

		#--------Load a (frozen) Tensorflow model and session into memory
    		self.detection_graph = tf.Graph()
    		with self.detection_graph.as_default():
      			od_graph_def = tf.GraphDef()
      			with tf.gfile.GFile(self.gpath, 'rb') as fid:
        			serialized_graph = fid.read()
        			od_graph_def.ParseFromString(serialized_graph)
        			tf.import_graph_def(od_graph_def, name='')

			self.sess = tf.Session(graph=self.detection_graph,config=config)
		

		# Define input and output Tensors for detection_graph
        	self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        	# Each box represents a part of the image where a particular object was detected.
        	self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        	# Each score represent  level of confidence for each of the objects.
        	# Score is shown on the result image, together with the class label.
        	self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        	self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        	self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

	
	rospy.logwarn("Done with Initiation ..")
        #pass
	

    def read_traffic_lights(self, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5, traffic_ligth_label=10):
    	im_width = self.image.shape[1]
	im_height = self.image.shape[0]
    	result = TrafficLight.UNKNOWN
	
    	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        	if (scores[i] > min_score_thresh and classes[i] == traffic_ligth_label):
            		ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            		(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            		
			crop_img = self.image[top:bottom, left:right, :]

			result = self.opencv_method(crop_img)
                		

    	return result

    
    def opencv_method(self, image):
	result = TrafficLight.UNKNOWN
	#output = self.image.copy()
	red = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	
	red1mask = cv2.inRange(red, self.lower_red_a, self.upper_red_a)
	
	
	red2mask = cv2.inRange(red, self.lower_red_b, self.upper_red_b)

	converted_img = cv2.addWeighted(red1mask , 1.0, red2mask, 1.0, 0.0)
	gblur_img = cv2.GaussianBlur(converted_img,(15,15),0)
	
	circles = cv2.HoughCircles(gblur_img, cv2.HOUGH_GRADIENT, 0.5, 41, param1=70, param2=30, minRadius=5, maxRadius = 150)

	found = False
	if circles is not None:
		result = TrafficLight.RED
	
	return result
        
     
    def rcnn_method(self):
	result = TrafficLight.UNKNOWN
	
	image_np_expanded = np.expand_dims(self.image, axis=0)
	(boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np_expanded})
	
	result = self.read_traffic_lights(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
       
		 
	return result	

    def get_classification(self,image,RCNN_CLASSIFIER):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	self.image = image
	
        #TODO implement light color prediction
	if(RCNN_CLASSIFIER is False):

		rospy.logwarn("Trying openCV method ..")
		result  = self.opencv_method(self.image)
		rospy.logwarn("opencv result is....%s",result)
		
	else:
		
		rospy.logwarn("Trying rcnn method ..")
		result  = self.rcnn_method()
		rospy.logwarn("RCNN result is....%s",result)
		
	return result
	
