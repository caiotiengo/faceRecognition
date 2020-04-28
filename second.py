# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 02:08:44 2020

@author: ct297154
"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

#carrega o model
model = model_from_json(open("fer.json", "r").read())
#carrega medidas
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:
    #captura o frame e retorna um boolean com valor e a imagem 
    ret,test_img=cap.read()
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#area imagem
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #Achar o maior INDEXED array
        max_index = np.argmax(predictions[0])

        emotions = ('Irritado', 'Enjoado', 'Confuso','Feliz', 'Triste', 'Surpreso', 'Neutro')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Fala comigo bb',resized_img)



    if cv2.waitKey(10) == ord('q'):#Aguardar o q para sair
        break

cap.release()
cv2.destroyAllWindows