# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
#sp_path - shape predictor path
#img_path - image path
sp_path="/run/media/estrossa/fast_ext/estrossa/zips/shape_predictor_68_face_landmarks.dat"		

img_path="/run/media/estrossa/fast_ext/estrossa/image/8.jpg"




# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,sp_path)
# ap.add_argument("-i", "--image", required=True,img_path)
# args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor(sp_path)

# load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
image = cv2.imread(img_path)
image = imutils.resize(image, width=500)
# cv2.imshow("img",image)
# cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1,cv2.LINE_AA)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	tx=1
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		cv2.putText(image, "{}".format(tx), (x - 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1,cv2.LINE_AA)
		tx+=1
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
