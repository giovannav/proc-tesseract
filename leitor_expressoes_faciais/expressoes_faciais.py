import numpy as np
from keras.models import load_model
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from time import sleep
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.experimental.set_visible_devices([], 'GPU')

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("model.h5")

emotion_labels = ['raiva', 'nojo', 'medo', 'feliz', 'neutro', 'triste', 'surpresa']

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Transforma o frame capturado em cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Encontra o rosto da pessoa
    faces = face_classifier.detectMultiScale(gray)

    # Percorre as coordena
    for (x, y, w, h) in faces:
        
        # Desenha um retângulo no rosto encontrodado
        # Top left: (x,y)
        # Bottom right: (x+w,y+h)
        cv.rectangle(frame, (x,y), (x+w,y+h),(255, 0, 255),2)
        
        # Recorta apenas o rosto identificado
        pre_predict = gray[y:y+h,x:x+w]
        
        # Redimensiona pra o tamanho aceito na rede
        pre_predict = cv.resize(pre_predict, (48, 48))

        # Verifica se a imagem não está vazia
        if np.sum([pre_predict])!=0:
            
            # Normaliza a image 0:255 -> 0:1
            face = pre_predict.astype('float')/255.0
            
            # Transforma em um array aceito pelo tf
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Faz o prediction
            prediction = model.predict(face)[0]
            
            # Seleciona da lista o prediction com maior valor
            label=emotion_labels[prediction.argmax()]
            
            # Escreve o resultado na imagem
            cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Nenhum rosto na imagem.', (30,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv.imshow('Leitor de expressoes', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()