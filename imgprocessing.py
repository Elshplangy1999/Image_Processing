# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:24:02 2022

@author: Emad Elshplangy
"""
from tkinter import *
from tkinter import ttk

import cv2 as cv # if not installed please consider running : pip install opencv-python
from PIL import ImageTk,Image # if not installed please consider running : pip install Pillow
import matplotlib # if not installed please consider running : pip install matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class AppGui:
    
    def __init__(self,root:Tk) :
        self.TARGET_IMG_PATH="Tree.png"
        self.RESULT_IMG_PATH="res.png"
        self.root=root
        self.resFrame=None
        self.configWindow(root)
        self.buildMethodSelectionFrame(root)
        self.buildOriginalImgViewr(root,self.TARGET_IMG_PATH)
    
    #configur window size and make it fixed.
    def configWindow(self,window:Tk):
        window.title("Histogram Equalizer")
        window.geometry('800x600')
        window.resizable(0,0)
    
    def buildMethodSelectionFrame(self,window):
        self.methodSelectFrame=LabelFrame(window,text="Equalization Method")
        methodSelected=IntVar()
        methodSelected.set(0)
        ttk.Radiobutton(self.methodSelectFrame,text="OpenCV Equalizaer",variable=methodSelected,value=1,command=self.performOpenCvHistEqualizer).pack(side=LEFT)
        ttk.Radiobutton(self.methodSelectFrame,text="Clahe Equalizer",variable=methodSelected,value=2,command=self.performClahe).pack(side=RIGHT)
        self.methodSelectFrame.pack(side=TOP,pady=20)


    def performOpenCvHistEqualizer(self):
        img = cv.imread(self.TARGET_IMG_PATH,0)
        equalized = cv.equalizeHist(img)
        cv.imwrite(self.RESULT_IMG_PATH,equalized)
        self.buildResultlImgViewr("OpenCV Equalizer",self.root,self.RESULT_IMG_PATH)


    def performClahe(self):
        img = cv.imread(self.TARGET_IMG_PATH,0)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(img)
        cv.imwrite(self.RESULT_IMG_PATH,equalized)
        self.buildResultlImgViewr("Clahe Equalizer",self.root,self.RESULT_IMG_PATH)
        

    def buildOriginalImgViewr(self,window,img):
        frame=LabelFrame(window,text="Original Image")
        #to view the image itself.
        im = Image.open(img).resize((300,200), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(im)
        Label(frame,image=self.photo).pack(side=TOP,pady=20)
        
        #to view image hisotgram
        cvImg = cv.imread(img,0)
        fig = Figure(figsize=(5,4), dpi=70)
        a = fig.add_subplot(111)
        a.hist(cvImg.flatten(),256,[0,256], color = 'r')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM)

        frame.pack(padx=20,side=LEFT)
    
    def buildResultlImgViewr(self,methodName,window,img):
        #to make sure no other result frame shown
        if(self.resFrame != None):
            for widget in self.resFrame.winfo_children():
                widget.destroy()
            self.resFrame.pack_forget()
        
        #view the result image
        self.resFrame=LabelFrame(window,text= methodName+" Image")
        im = Image.open(img).resize((300,200), Image.ANTIALIAS)
        self.resPhoto = ImageTk.PhotoImage(im)
        Label(self.resFrame,image=self.resPhoto).pack(side=TOP,pady=20)
        
        #view histogram of result image
        cvImg = cv.imread(img,0)
        fig = Figure(figsize=(5,4), dpi=70)
        a = fig.add_subplot(111)
        a.hist(cvImg.flatten(),256,[0,256], color = 'b')
        canvas = FigureCanvasTkAgg(fig, master=self.resFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM)

        self.resFrame.pack(padx=20,side=RIGHT)



if __name__ == '__main__':
    root=Tk()
    AppGui(root)
    root.mainloop()

