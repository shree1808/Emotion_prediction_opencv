import cv2
import numpy as np
import keras
from keras.models import load_model
import logging

logging.basicConfig(
    filename= 'logs/Opencv_deployment_error_logs.log',
    level= logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s'
)

# Imported the MobileNet50 // # Input shape -> (224,224,3)
model = load_model('my_model.h5')

emotion_labels = ['Sad', 'Fearful', 'Neutral', 'Disgusted', 'Happy', 'Surprised', 'Angry']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Importing the face classifier
face_classifier = cv2.CascadeClassifier('Harcascades\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_detect = face_classifier.detectMultiScale(rgb_frame, 1.3, 5)

        for (x,y,w,h) in face_detect:

            face = face_detect[y:y+h, x:x+w]
            face = cv2.resize(face, (224,224))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis = 0)  # Making the input image shape -> (1,224,224,3)

            # Predicting the emotion
            emotion_prediction = model.predict(face)
            max_index = np.argmax(emotion_prediction)
            emotion = emotion_labels[max_index]

            # Displaying the results
            cv2.Rectangle(frame, (x,y), (x+w , y+h), (255,0,0), 2)
            cv2.putText(frame, emotion , (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9 , (36,255,12), 2)
            
        
        cv2.imshow('Real Time Emotion Detection Application', frame)
        logging.info(f'The prediction was done!')
    
    else:
        print('Failed to Capture frame!!')

        if cv2.waitKey(1) & 0xff == ord('q'):    
            break

cap.release()
cv2.destroyAllWindows()

