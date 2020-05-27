# import the necessary packages
from __future__ import print_function
from __future__ import division
# from imutils import face_utils
import numpy as np
import argparse
# import imutils
import dlib
import cv2
import os
import time
from FaceAlign import FaceAligner


def dlibVect_to_numpyNDArray(vector):
    array = np.zeros(shape=128)
    for i in range(0, len(vector)):
        array[i] = vector[i]
    return array


imshows=1


sp_path="/run/media/estrossa/fast_ext/estrossa/zips/5landmarks.dat"		


img_path="/run/media/estrossa/fast_ext/estrossa/image/11.jpg"
network_model="/run/media/estrossa/fast_ext/estrossa/face_verification_experiment/model/resnet.dat"


print(img_path)
arra_path='dest.npy'
comp=0
if(len(arra_path)>0):
    comp=1
    dest=np.load(arra_path)
    print(dest[0],dest[1],dest.shape)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(sp_path)
recognizer= dlib.face_recognition_model_v1(network_model)

fa=FaceAligner(predictor)

st=time.time()
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for rect in rects:
    

    faceAligned = fa.align(image, gray, rect)

    
   
    if(imshows):cv2.imshow("img",faceAligned)
    
    scores=recognizer.compute_face_descriptor(faceAligned)      #score in dlib vector
    np_score=dlibVect_to_numpyNDArray(scores)

    if(comp):
        print("comparision ")
        print(np.linalg.norm(dest-np_score))
    else:
        np.save('dest.npy',np_score)
        print("saves as dest.npy")
    
    # cv2.imshow('img',out)
    if(imshows):cv2.waitKey(0)
    
print("time it took to run the loop ",time.time()-st)