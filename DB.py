import sqlite3
from sqlite3 import Error as sqlerror
import numpy as np
import multiprocessing as mp
import time
import logging
import sys
import os





class details:
    def __init__(self,rollNo,firstname,lastname):
        self.rollNo=rollNo
        self.firstname=firstname
        self.lastname=lastname

class connectDB:
    def __init__(self):
        self.file='facedata.db'
        try:
            if self.isSQLite3():
                self.sqliteConnection = sqlite3.connect(self.file)
                self.sqliteConnection.text_factory=str
                print("Connected to SQLite")

            else:
                raise DBNotExist    
        except DBNotExist:
            print("DBNotExist")
            raise DBNotExist
    
    
    def rollNoCheck(self,rollNo):
        cursor = self.sqliteConnection.cursor()
       
        query='select exists(select 1 from student where rollNo=? collate nocase) limit 1'
        # 'query' RETURNS 1 IF USERNAME EXISTS OR 0 IF NOT, AS INTEGER(MAYBE). 'collate nocase'= CASE INSENSITIVE, IT'S OPTIONAL
        check=cursor.execute(query,(rollNo,)) 
        
        if check.fetchone()[0]==0:
            cursor.close()
            return True
        else:
            cursor.close()
            return False
       
    
    def isSQLite3(self):
        from os.path import isfile, getsize

        if not isfile(self.file):
            raise DBNotExist
        if getsize(self.file) < 100: # SQLite database file header is 100 bytes
            raise DBNotExist

        with open(self.file, 'rb') as fd:
            header = fd.read(100)

        return header[:16] == 'SQLite format 3\x00'
    
    
    def convertToBinaryData(self,np_array):
        #Convert digital data to binary format
        return np_array.tobytes()

    


    def insertBLOB(self,rollNo, FirstName, LastName, np_array):
        try:
            
            
            sqliteConnection=self.sqliteConnection
            cursor = sqliteConnection.cursor()
            
            
            sqlite_insert_blob_query = """ INSERT INTO student
                                    (rollNo, FirstName, LastName, embedd) VALUES (?, ?, ?, ?)"""

            
            print np_array[:5]
            arrayx = self.convertToBinaryData(np_array)

            print str(type(arrayx))+' hi printing the type of np array'
            # Convert data into tuple format
            data_tuple = (rollNo, FirstName, LastName, buffer(arrayx),)
            cursor.execute(sqlite_insert_blob_query, data_tuple)
            sqliteConnection.commit()
            print("Name and array inserted successfully as a BLOB into a table")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert blob data into sqlite table", error)
        



    def blobtoNP(self,barray):
        out=np.frombuffer(barray, dtype=float)
        return out

    def find_face_match(self,cur_array):
        
        cursor = self.sqliteConnection.cursor()


        detect = False
        rollNo=0
        FirstName=''
        LastName=''
           

        sql_fetch_blob_query = """SELECT * from student"""
        cursor.execute(sql_fetch_blob_query,)
        record = cursor.fetchall()
        
        for row in record:
            print("Id = ", row[0], "FirstName = ", row[1])
            rollNo = row[0]
            FirstName = row[1]
            LastName  = row[2]
            barray=row[3]
            
            #decoding the values
            
            np_array = self.blobtoNP(barray)
            print 'printing the array from db -----------------------------------'
            
            # print(len(np_array))
            
            finalscore = np.linalg.norm(cur_array-np_array)
            print "finalscore %d" % finalscore
            if(finalscore <= 0.6 ):
                detect=1
                break

        
        cursor.close()
        if detect:
            print("some detecton done ")
            return rollNo,FirstName,LastName
        else:
            return False,False,False



    def db_multi_thead_run(self,inqueue,outqueue):
        
        db_proc = mp.Process(target=self.db_multi_thead,args=(inqueue,outqueue))
        print("db thread started..........")
        db_proc.daemon=True
        db_proc.start()








    def db_multi_thead(self,inqueue,outqueue):
        

        while True:
            time.sleep(0.06)         
            while(inqueue.qsize()==0):
                time.sleep(1)
                print("sleeping")
            
            array = inqueue.get(block=True,timeout=2)
            if(array is None):
                return 
            # print("hello 1"+str(self.inqueue.qsize()))
            # if():
            #     self.stop()
                # return 
            # sys.stdout.flush()
            rollno,firstname,lastname=self.find_face_match(array)
            
            if(rollno):
                outqueue.put( details( rollno, firstname, lastname ) )

            sys.stdout.flush()
            # print type(np_score)
            # np_score=vals



#make the closing connection function
    def close_connection(self):
            if (self.sqliteConnection):
                self.sqliteConnection.close()
                print("the sqlite connection is closed")

    def __del__(self):
        # body of destructor
        if (self.sqliteConnection):
                self.sqliteConnection.close()
                print("the sqlite connection is closed")



class DBNotExist(Exception):
    """Raised when the db file does not exist"""
    pass




# db=connectDB()
# print(db.find_face_match(np.arange(1,129)))