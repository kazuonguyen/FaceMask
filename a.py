
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import numpy as np
import keras
import keras.backend as k
from keras.models import Sequential,load_model
from keras.optimizers import Adam
import cv2
import datetime
import tensorflow
np.set_printoptions(suppress=True)


model = tensorflow.keras.models.load_model('keras_model.h5')

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while cap.isOpened():
    ref,frame=cap.read()
    face=face_cascade.detectMultiScale(frame,1.1, 4)
    for(x,y,w,h) in face:
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open('temp.jpg')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        
        data[0] = normalized_image_array
        pred=model.predict(data)[0][0]
        if pred>0.2:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(frame,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet=str(datetime.datetime.now())
        cv2.putText(frame,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('Detect',frame)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
