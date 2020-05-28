# import the necessary packages
from __future__ import print_function
from __future__ import division
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from collections import OrderedDict
from skimage import color


from imutils import face_utils 
#sp_path - shape predictor path
#img_path - image path
sp_path="/run/media/estrossa/fast_ext/estrossa/zips/shape_predictor_5_face_landmarks.dat"		

img_path="/run/media/estrossa/fast_ext/estrossa/image/11.jpg"
network_model="/run/media/estrossa/fast_ext/estrossa/face_verification_experiment/model/resnet.dat"
# net = caffe.Classifier(network_def, network_model,None,raw_scale=1)

print(img_path)
arra_path='dest.npy'
comp=0
if(len(arra_path)>0):
    comp=1
    dest=np.load(arra_path)
    print(dest[0],dest[1],dest.shape)
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
recognizer= dlib.face_recognition_model_v1(network_model)

fa=face_utils.FaceAligner(predictor,desiredFaceWidth=512)
# load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
image = cv2.imread(img_path)
image = imutils.resize(image, width=500)
# cv2.imshow("img",image)
# cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)

    # print("hi-----------------------------------------------------------------------------------------")
    # print(faceAligned.shape)

    
    out=cv2.resize(faceAligned,(150,150))
    print(out.shape)
    
    # cv2.imshow('img',out)
    # cv2.waitKey(0)
    # out=cv2.dnn.blobFromImage(out,scalefactor=(1/255))
    # out=out[::]
    # print(out.shape)
    # out=out[np.newaxis,:,:]
    # out=out[np.newaxis,np.newaxis,:,:]
    # print(out.shape)

    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # #transformer.set_transpose('data', (2,0,1))
    # image = out[:, :, ::-1]
    # image = color.rgb2gray(image).reshape(128, 128, 1)
    # print(image.shape)
    # transformed_image = transformer.preprocess('data', image)
    

    # out=out/255

    
    
   
    # net.blobs['data'].data[...] = transformed_image
    # net.blobs['data'].data[...] =out/255
    
    
    # print(out.shape)
    # display the output images
    # cv2.imshow("Original", faceOrig)
    # cv2.imshow("Aligned", faceAligned)
    # cv2.imwrite("henlo.jpeg",faceAligned)
    # cv2.waitKey(0)

    ##Recognition
    # print(out.shape)
    # for xt in out[2]:
    #     print(xt[0],end=' ')


    # scores = net.predict(out, oversample=False)
    # scores=net.forward()
    # blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    #shp = blobs[layer_name].shape
    
    print("Hiiiiiiiiiiiiiiiiiiiiii=================================================")
    # print(scores['prob'].shape)
    scores=scores['prob']
    for i in range(10):
        print(scores[0][i] ,end=' ')
    print("=============================================================================") 

    if(comp):
        print(np.linalg.norm(dest-scores[0]))
    else:
        np.save('dest.npy',scores[0])
        print("saves as dest.npy")





#  print img_path
#     print 'hello'
#     img = caffe.io.load_image(img_path,color=False)
# ##modifications to handle image properly
#     img=caffe.io.resize_image(img,(128,128))


# #end
#     net = caffe.Classifier(network_def, network_model,None,raw_scale=1)
    #net.set_phase_test()
    # net.set_mode_cpu()
    # net.set_device(2)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
    #net.set_mean('data', mean_file)
    # net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    #net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
    # net.set_input_scale('data', 1)
    
    #sio.savemat(save_path, blobs)
   