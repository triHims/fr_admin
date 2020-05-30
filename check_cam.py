import cv2
import time
from imutils import face_utils
import numpy as np







from threading import Thread

from Queue import Queue
from Queue import Empty






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
            time.sleep(.06)
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
            print("Queue Empty Ending")
            return not self.stopped
        else:
            return True
            
        
        

    def stop(self):
        self.stopped=True
        self.stream.release()
        self.Q.queue.clear()








    
    
    



    

vid=ThreadStream(0,100)




vid.start()
time.sleep(1.0)

tx=time.time()
cnt=0




while(vid.more()):
    
    tx=time.time()
    frame = vid.read()
    # time.sleep(0.2)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        

        # print(rects.shape)
    
    
    if(time.time()-tx < 0.06 ): time.sleep( 0.06-(time.time()-tx))





    cv2.imshow('Capturing Video',frame)

   



    print("time taken "+str(time.time()-tx))
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        vid.stop()

    

    cnt+=1
    cnt%=100000
        
vid.stop()

