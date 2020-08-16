# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:53:20 2020

@author: Wenjun Zeng
"""
import numpy as np
import os
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
from numpy import savetxt
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from numpy import asarray
from numpy import savetxt


def printPredict(InputFile, OutputFile):
    X = np.genfromtxt(InputFile, delimiter=',')
    y = np.genfromtxt(OutputFile, delimiter=',')
    #Normalization
    X_scale = preprocessing.scale(X)
    print (X_scale)

    X_train = X_scale[:19]
    X_test = X_scale[:31]

    y_train = y[:19,:3]
    y_test= y[:31,:3]

    BaseModel = load_model('my_model.h5')
    training_epochs = 2000 # epochs
    batch_size = 15#number of batch
    BaseModel.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])#mape
    BaseModel.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs, validation_data = (X_test, y_test))
    predict = BaseModel.predict(X_test)
    print (predict)
    data = asarray(predict)
    # save numpy array as csv file
    savetxt('predictOutput.csv', data, delimiter=',')
    os.startfile('predictOutput.csv')
    # save ANN model as h5 file
    #BaseModel.save('my_model.h5')

def printAccuracy(InputFile, OutputFile):
    X = np.genfromtxt(InputFile, delimiter=',')
    y = np.genfromtxt(OutputFile, delimiter=',')
    #Normalization
    X_scale = preprocessing.scale(X)
    print (X_scale)

    X_train = X_scale[:19]
    X_test = X_scale[:31]

    y_train = y[:19,:3]
    y_test= y[:31,:3]

    r=[0,0,0]
    training_epochs = 2000 # epochs
    batch_size = 15#number of batch
    BaseModel = load_model('my_model.h5')
    BaseModel.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs, validation_data = (X_test, y_test))

    predict = BaseModel.predict(X_test)
    strList = "\t"
    for i in range (np.size(predict, 0)):#np.size(predict, 0)
        for j in range (np.size(predict, 1)):#np.size(predict, 1)
            r[j]+=abs((predict[i,j]-y_test[i,j])/y_test[i,j])
    for i in r:
        print ((i/np.size(predict, 0)))
        strList += str((i/np.size(predict, 0)))
        strList +="\n\t"
    pred_acc = r2_score(y_test, predict)
    StrR2 = str(pred_acc)

    messagebox.showinfo( "Output","R2: \n\t"+StrR2+"\n"+"Errors:"+"\n"+strList)

def plotLoss(InputFile, OutputFile):
    #Extracct the data
    X = np.genfromtxt(InputFile, delimiter=',')
    y = np.genfromtxt(OutputFile, delimiter=',')
    #Normalization
    X_scale = preprocessing.scale(X)
    print (X_scale)

    X_train = X_scale[:19]
    #X_train = X_scale
    X_test = X_scale[:31]

    y_train = y[:19,:3]
    #y_train_2 = y[:15,2]
    y_test= y[:31,:3]
    #y_test_2 = y[15:19,2]
    training_epochs = 2000 # epochs
    batch_size = 15#number of batch
    BaseModel = load_model('my_model.h5')
    BaseModel.compile(loss='mean_squared_error', optimizer='adam',metrics=['mape'])
    history = BaseModel.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs, validation_data = (X_test, y_test))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
#plot Accuracy
def plotAcc(InputFile, OutputFile):

    X = np.genfromtxt(InputFile, delimiter=',')
    y = np.genfromtxt(OutputFile, delimiter=',')
    #Normalization
    X_scale = preprocessing.scale(X)
    print (X_scale)

    X_train = X_scale[:19]
    #X_train = X_scale
    X_test = X_scale[:31]

    y_train = y[:19,:3]
    #y_train_2 = y[:15,2]
    y_test= y[:31,:3]
    #y_test_2 = y[15:19,2]
    BaseModel = load_model('my_model.h5')
    training_epochs = 2000 # epochs
    batch_size = 15#number of batch
    #print(standardized_X)
    BaseModel.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    history = BaseModel.fit(X_train, y_train,validation_split=0.25, batch_size=batch_size, epochs=training_epochs, validation_data = (X_test, y_test), verbose=1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#User Interface
list = []
h = []
window = Tk()
style = ttk.Style(window)
style.theme_use("clam")

window.title("Welcome to Artificial Neural Network simulator")

window.geometry('500x300+100+200')

lbl3 = Label(window, text="Please choose the function:")
lbl3.grid(column=1, row=250)
'''
txt = Entry(window,width=10)
txt.grid(column=20, row=30)

txtOut = Entry(window,width=10)
txtOut.grid(column=20, row=50)
'''
def callbackFunc(event):
     print(cbox.current(), cbox.get())

cbox = ttk.Combobox(window, values=["Save predicted output",
                                    "Display errors and R2",
                                    "Plot loss history",
                                    "Plot accuracy history"], state='readonly')
cbox.grid(column=2, row=300)
cbox.set("select")
cbox.bind("<<ComboboxSelected>>", callbackFunc)

def open_file_Input():
    repIn = filedialog.askopenfilenames(
    	parent=window,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
    		("CSV", "*.csv"),
    		("All files", "*")])
    textIn = "Input file: "+str(repIn)
    lbl2 = Label(window, text= repIn[0])
    lbl2.grid(column=5, row=100)
    list.append(repIn[0])
    #print(InputFilePath)


def open_file_Output():
    repOut = filedialog.askopenfilenames(
    	parent=window,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
    		("CSV", "*.csv"),
    		("All files", "*")])
    textOut = "Output file: "+str(repOut)
    lbl = Label(window, text=repOut[0])
    lbl.grid(column=5, row=200)
    list.append(repOut[0])
    #print(OutputFilePath)

btn2 = Button(window, text="Browse Input file", command=open_file_Input).grid(row=100, column=2, padx=4, pady=4, sticky='ew')
btn3 = Button(window, text="Browse Output file", command=open_file_Output).grid(row=200, column=2, padx=4, pady=4, sticky='ew')

def clicked():
    InputFile = list[0]
    OutputFile = list[1]

    if (cbox.get() == "Save predicted output"):
        printPredict(InputFile, OutputFile)

    elif(cbox.get() =="Display errors and R2"):
        printAccuracy(InputFile, OutputFile)

    elif(cbox.get() =="Plot loss history"):
        plotLoss(InputFile, OutputFile)

    elif (cbox.get() =="Plot accuracy history"):
        plotAcc(InputFile, OutputFile)

    #messagebox.showinfo( "Exit", "Program ends")


btn = Button(window, text="Confirm", command=clicked)
btn.grid(column=2, row=400)

window.mainloop()
