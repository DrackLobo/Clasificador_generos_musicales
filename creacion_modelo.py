import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers.legacy import Adam

from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#C1
datos = "./genres_original"
#classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

classes = ['blues', 'classical']
#C2
def procesarDatosDirectorio(data_dir,classes,target_shape=(150,150)):
    data=[]
    labels=[]

    for i_class,class_name in enumerate(classes):
        class_dir = os.path.join(data_dir,class_name)
        print("Processing--",class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir,filename)
                audio_data,sample_rate = librosa.load(file_path,sr=None)
                
                duracion_segmento = 4
                duracion_superposicion = 2
                
                muestras_segmento = duracion_segmento * sample_rate
                muestras_superposicion = duracion_superposicion * sample_rate
                
                
                num_segmentos = int(np.ceil((len(audio_data)-muestras_segmento)/(muestras_segmento-muestras_superposicion)))+1
                
                
                for i in range(num_segmentos):
                   
                    inicio = i*(muestras_segmento-muestras_superposicion)
                    fin = inicio+muestras_segmento
                    #
                    segmento = audio_data[inicio:fin]
                    #Melspectrogram part
                    mel_spectrogram = librosa.feature.melspectrogram(y=segmento,sr=sample_rate)
                    #Resize matrix based on provided target shape
                    mel_spectrogram = resize(np.expand_dims(mel_spectrogram,axis=-1),target_shape)
                    #Append data to list
                    data.append(mel_spectrogram)
                    labels.append(i_class)
    #Return
    return np.array(data),np.array(labels)

#c3
data,labels = procesarDatosDirectorio(datos,classes)
#Errores al cargar jazz.00054.wav

#c4
print(data.shape) 
print(labels.shape)

#c5
labels = to_categorical(labels,num_classes = len(classes)) # Converting labels to one-hot encoding
print(labels)

#c6
print(labels.shape)

X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2,random_state=42)


model = tf.keras.models.Sequential()
print(X_train[0].shape)

#c9
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=X_train[0].shape))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

#c10
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

#c11
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

#c12
model.add(Dropout(0.3))

#C13
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

#c14
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

#c15
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=1200,activation='relu'))
model.add(Dropout(0.45))

#c16
model.add(Dense(units=len(classes),activation='softmax'))

#c17
model.summary()

#c18

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

#c19
training_history = model.fit(X_train,Y_train,epochs=30,batch_size=32,validation_data=(X_test,Y_test))

model.save("Trained_model.h5") #Windows  

