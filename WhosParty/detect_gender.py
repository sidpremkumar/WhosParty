# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

class DetectGender:
	model_path = 0;

	def __init__(self):
		"""
		Constructor to download pre-trained model
		"""
		# download pre-trained model file (one-time download)
		dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
		self.model_path = get_file("gender_detection.model", dwnld_link,
		                     cache_subdir="pre-trained", cache_dir=os.getcwd())

	def classifyImage(self, file_path):
		"""
		Function to classify an image and return classification
		:param String file_path: Path to the image
		Returns list [man confidence, women confidence]
		"""
		# read input image
		image = cv2.imread(file_path)

		if image is None:
		    raise Exception("Could not read input image")

		# load pre-trained model
		model = load_model(self.model_path)

		# detect faces in the image
		face, confidence = cv.detect_face(image)

		classes = ['man','woman']

		# loop through detected faces
		try:
			for idx, f in enumerate(face):
			    # get corner points of face rectangle
			    (startX, startY) = f[0], f[1]
			    (endX, endY) = f[2], f[3]

			    # draw rectangle over face
			    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

			    # crop the detected face region
			    face_crop = np.copy(image[startY:endY,startX:endX])

			    # preprocessing for gender detection model
			    face_crop = cv2.resize(face_crop, (96,96))
			    face_crop = face_crop.astype("float") / 255.0
			    face_crop = img_to_array(face_crop)
			    face_crop = np.expand_dims(face_crop, axis=0)

			    # apply gender detection on face
			    conf = model.predict(face_crop)[0]
			    return conf, classes
		except Exception as e:
			raise e


		#
		# # display output
		# cv2.imshow("gender detection", image)
		#
		# # press any key to close window
		# cv2.waitKey()
		#
		# # save output
		# cv2.imwrite("gender_detection.jpg", image)
		#
		# # release resources
		# cv2.destroyAllWindows()
