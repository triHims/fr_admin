import Tkinter as tk
import ttk as ttk

from functools import partial

def validateSubmit(tkWindow,rollno, fname, lname,dbase, errorstr):
    
    first = fname.get()
    last = lname.get()
    rollNo=rollno.get()
    if( first and rollNo and len(str(rollNo))>8 ):
        if(dbase.rollNoCheck(rollNo)):
            tkWindow.destroy()
        else:
            errorstr.set('Roll number Exists')
    else :
        errorstr.set('name or Roll Number cannot be Empty')
    


    # if('''some condtion to varify rollnumer is unique''')
    # {
    #     '''enroll logic'''
    # }
    
    return


class dmy:
    def rollNoCheck(self,x):
        return True




def guiEnroll(dbase):
#window
    
    tkWindow = tk.Tk()  
    tkWindow.geometry('400x150')  
    tkWindow.title('Enroll From')

    #username label and text entry box
    rollnoLabel = ttk.Label(tkWindow, text="RollNumber").grid(row=0, column=0)
    rollno = tk.StringVar()
    rollnoEntry = ttk.Entry(tkWindow, textvariable=rollno).grid(row=0, column=1)
    ttk.Label(tkWindow, text="(Numbers only)").grid(row=0, column=2)  

    #password label and password entry box
    fnameLabel = ttk.Label(tkWindow,text="FirstName").grid(row=1, column=0)  
    fname = tk.StringVar()
    fnameEntry = ttk.Entry(tkWindow, textvariable=fname).grid(row=1, column=1)

    lnameLabel = ttk.Label(tkWindow,text="LastName").grid(row=2, column=0)  
    lname = tk.StringVar()
    lnameEntry = ttk.Entry(tkWindow, textvariable=lname).grid(row=2, column=1)  
    
    
    errorstr=tk.StringVar()
    errorLabel = ttk.Label(tkWindow, textvariable=errorstr, foreground='red').grid(row=3, column=1, columnspan=4, sticky=(tk.W) )

    vs = partial(validateSubmit,tkWindow, rollno, fname, lname, dbase,errorstr)

    #login button
    submitButton = ttk.Button(tkWindow, text="Submit", command=vs).grid(row=5, column=2)  
    for child in tkWindow.winfo_children(): child.grid_configure(padx=5, pady=5)
    tkWindow.mainloop()


    return rollno.get(),fname.get(),lname.get()
        


# r,f,l=guiEnroll(dmy())

