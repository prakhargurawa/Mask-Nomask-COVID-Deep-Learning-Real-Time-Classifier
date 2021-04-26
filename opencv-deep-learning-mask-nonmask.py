# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:14:07 2021

@author: prakh
"""

# import the needed packages
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import seaborn as sn
import pandas as pd
import cv2
from keras.models import load_model
from pygame import mixer
from PIL import Image

# Model structure lossly inspired from research paper : https://link.springer.com/content/pdf/10.1007%2F978-3-030-66665-1_6.pdf
base_model =  keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)  # to reduce overfitting
x = keras.layers.Dense(128, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)

full_model = Model(inputs=base_model.input, outputs=predictions)
full_model.load_weights("vgg2model.h5")


mixer.init()
sound = mixer.Sound('alarm.wav')
bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
cv2.namedWindow("COVID Mask Detection Video Feed")
webcam = cv2.VideoCapture(0) 

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    rval, im = webcam.read()
    im=cv2.flip(im,1,1)
    
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
 
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f] 
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=full_model.predict(reshaped)
        if result[0][0] > result[0][1]:
            percent = round(result[0][0]*100,2)
            print("no mask")
            sound.play()
        else:
            print("mask")
            percent = round(result[0][1]*100,2)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label] + " " + str(percent) + "%", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    if im is not None:   
        cv2.imshow('COVID Mask Detection Video Feed', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Stop video
webcam.release()