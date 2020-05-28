import cv2
import time
# from __future__ import print_function
import sys
from imutils import face_utils
import numpy as np
import argparse


from FaceAlign import FaceAligner
from Gui import guiEnroll


import dlib
import cv2
import os

import misc
from threading import Thread

from Queue import Queue
from Queue import Empty
import multiprocessing as mp
import MPQueue
from DB import connectDB
import logging

os.environ['GLOG_minloglevel'] = '50'

#setup parser
parser = argparse.ArgumentParser("Start the Attendence system in Recognize mode")
group = parser.add_mutually_exclusive_group()
group.add_argument("-r", "--recognize", action="store_true",help="Start in recognize (Default)")
group.add_argument("-e", "--enroll", action="store_true",help="Start in enroll mode")
args=parser.parse_args()
#flag to toggle inroll and recognize

flag_rec=1

if(args.enroll):
    flag_rec=0
    print("Enroll mode Set")



#model declarations 
sp_path="references/5landmarks.dat"		
network_model="references/resnet.dat"	







class dimms:
    def __init__(self,image,gray,rect):
        self.image=image
        self.gray=gray
        self.rect=rect

class ThreadStream:
    def __init__(self,path=0,queueSize=128):
        self.stream=cv2.VideoCapture(path)
        self.stopped=True
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        self.stopped=False
        t=Thread(target=self.update,args=())
        t.daemon=True
        t.start()
        return self

    def update(self):
        
        while True:

            if(self.stopped):
                return
            if not self.Q.full():
                success,frame = self.stream.read()

                if(not success):
                    self.stop()
                    return 




                scale_percent = 50

                #calculate the 50 percent of original dimensions
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)

                # dsize
                dsize = (width, height)

                # resize image
                frame=cv2.resize(frame,dsize)
                self.Q.put(frame)
            else:
                print("queue full")



    def read(self):
        return self.Q.get(block=True,timeout=10)

    def  more(self):
        if(self.Q.qsize() <= 0 ): 
            print("Queue Empty")
            return not self.stopped
        else:
            return True
            
        
        

    def stop(self):
        self.stopped=True
        self.stream.release()
        self.Q.queue.clear()







class FaceRecognize:
    def __init__(self,model_path,sp_path,queueSize,inqueue,outqueue):
        self.predictor = dlib.shape_predictor(sp_path)
        self.recognizer= dlib.face_recognition_model_v1(network_model)
        self.fa=FaceAligner(self.predictor)
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.t=mp.Process(target=self.Facerec,args=())
        
        
        
    


    def Alignment(self,image,gray,rect):
        return self.fa.align(image,gray,rect)
    def recognize(self,image):
        return self.recognizer.compute_face_descriptor(image)
    
    



    def start(self):
        self.stopped=False
        
        # time.sleep(2)
        self.t.daemon=True
        self.t.start()
        return self



    #mp rec function    
    def Facerec(self):
        
        



        while True:
            
            
            
            while(self.inqueue.qsize()==0):
                time.sleep(1)
                print("sleeping")
            
            dimm = self.inqueue.get(block=True,timeout=2)
            
            
            if(dimm is None):  #before dying kill outqueue
                try:
                    self.outqueue.put(None)
               
                except AssertionError:
                    return
                return 



            print("hello 1"+str(self.inqueue.qsize()))
            # if():
            #     self.stop()
                # return 
            sys.stdout.flush()
            aligned=self.Alignment(dimm.image,dimm.gray,dimm.rect)
            
            vals=self.recognize(aligned)
            np_score=misc.dlibVect_to_numpyNDArray(vals)
            # print type(np_score)
            # np_score=vals


                
            self.outqueue.put(np_score)
            # print(np_score[:5])



    def stop(self):
        
        
        try:
            self.inqueue.put(None)
            self.inqueue.close()
            self.outqueue.close()
            self.t.join()
        except AssertionError:
            return



def fr_read(outqueue):
        try:
            return outqueue.get(block=True,timeout=0.00005 )
        except Empty:
            return []


def db_read(outqueue):
    try:
        return outqueue.get(block=True,timeout=0.00005 )
    except Empty:
        return False
    # def  more(self):
    #     if(self.Q.qsize() <= 0 ): 
    #         print("Queue Empty")
    #         return not self.stopped
    #     else:
    #         return True
    
def fr_put(inqueue,image,gray,rect):
        t=dimms(image,gray,rect)
        inqueue.put(t)
        return
        

    


                
            
            # print(np_score[:5])
    


# predictor = dlib.shape_predictor(args["shape_predictor"])




# load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image


# loop over the face detections
if not flag_rec:
    db=connectDB()
    rollno,firstname,lastname=guiEnroll(db)
    



    vid=ThreadStream(0,100)
    inqueue=MPQueue.Queue(maxsize=20)
    outqueue=MPQueue.Queue()
    fcr=FaceRecognize(network_model,sp_path,512,inqueue,outqueue)

    vid.start()
    time.sleep(1.0)
    fcr.start()
    tx=time.time()
    cnt=0
    detector = dlib.get_frontal_face_detector()
    
    
    to_enroll=[]


    while(vid.more()):
        tx=time.time()
        frame = vid.read()
        # time.sleep(0.2)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(not cnt%5): 
            rects = detector(gray  , 1)
            # print("fifth")
            for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
                fr_put(inqueue, frame, gray , rect )
            

            # print(rects.shape)
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame,"press space to enroll , q to exit",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

                
        
        
        if(time.time()-tx < 0.105 ): time.sleep( 0.105-(time.time()-tx))





        cv2.imshow('Capturing Video',frame)

        out=np.array(fr_read(outqueue))
        
        to_enroll = out if (len(out) > 0 )  else to_enroll
        # print out.dtype
        print(to_enroll[:5])
        print("time taken "+str(time.time()-tx))
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            vid.stop()
            fcr.stop()
        if(cv2.waitKey(1) & 0xFF == ord(' ')):
            if( len(to_enroll) > 0 ):
                db.insertBLOB(rollno,firstname,lastname,to_enroll)
                cv2.destroyAllWindows()
            vid.stop()
            fcr.stop()

        cnt+=1
        cnt%=100000
            
    vid.stop()
    fcr.stop()





else:


    vid=ThreadStream(0,100)
    inqueue=MPQueue.Queue(maxsize=20)
    midqueue = MPQueue.Queue(maxsize=1024)
    outqueue = MPQueue.Queue()

    fcr=FaceRecognize(network_model,sp_path,512,inqueue,midqueue)
    # mp.log_to_stderr(logging.Error)
    db=connectDB()
    db.db_multi_thead_run(midqueue, outqueue)

    vid.start()
    time.sleep(1.0)
    fcr.start()
    tx=time.time()
    cnt=0
    detector = dlib.get_frontal_face_detector()
    
    
    lname=''
    # lrollno=''


    while(vid.more()):
        
        tx=time.time()
        frame = vid.read()
        # time.sleep(0.2)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(not cnt%5): 
            rects = detector(gray  , 1)
            # print("fifth")
            for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
                fr_put(inqueue, frame, gray , rect )
            

            # print(rects.shape)
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2) 
            cv2.putText(frame,lname,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
            cv2.putText(frame,"q to exit",(50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                
        
        
        if(time.time()-tx < 0.105 ): time.sleep( 0.105-(time.time()-tx))





        cv2.imshow('Capturing Video',frame)

        out=db_read(outqueue)
        sys.stdout.flush()
        if(out is not False):
            print 'the rollno Firstname and lastname are %s %s %s' %(out.rollNo, out.firstname, out.lastname )
            lname=out.firstname
            # lrollno=out.rollNo



        print("time taken "+str(time.time()-tx))
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            vid.stop()
            fcr.stop()
        

        cnt+=1
        cnt%=100000
            
    vid.stop()
    fcr.stop()
