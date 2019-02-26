#!/usr/bin/env python
# coding: utf-8




from PIL import Image
import os, glob, sys, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import Activation, BatchNormalization, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import losses
from keras import backend as K 
import matplotlib.pyplot as plt
import math
from keras.optimizers import SGD, Adam
from keras import metrics
from keras import models, layers, optimizers  
  


img_dir = './dataset'
categories = ['train', 'validation']
np_classes = len(categories)

image_w = 64
image_h = 64


np.random.seed(0)
pixel = image_h * image_w * 3

X = []
y = []
filenames = []
for idx, cat in enumerate(categories):
    img_dir_detail = img_dir + "/" + cat
    files = glob.glob(img_dir_detail+"/*/"+ "*.jpg")


    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            
            filenames.append(f)
            file_token=filenames[i].split("_")[1]    
            n = float(file_token)
            n = np.log10(n)

            X.append(data)
            y.append(n)
            
            if i % 1000 == 0:
                print(cat, " :\t", filenames[i]+ "  \t", y[i])    
                
        except:
            print(cat, str(i)+" 번째에서 에러 ")
            
X = np.array(X)
Y = np.array(y, dtype=np.int64) # 명시적으로 자료타입 알려줌. 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./numpy_data/binary_image_data.npy", xy)

X_train, X_test, y_train, y_test = np.load('./numpy_data/binary_image_data.npy')

X_train = X_train.astype('float32') / 255  
X_test = X_test.astype('float32') / 255

print(X_train.shape)
print(X_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))


Y_train = Y_train.astype('float32') / 255 
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32') / 255
Y_test = Y_test.astype('float32')

droprate=0.25


model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(64,64,3), activation="relu")) 
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))  
model.add(Flatten())
model.add(Dense(256, activation="relu")) # 256 임의의 수. 256개 입력받아 1개의 출력. 여러 수치 시도해 볼 것!
model.add(BatchNormalization())
model.add(Dense(1))
 
def rmsle(y_test, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred) - K.log(y_test))))

model.compile(loss= 'mean_squared_error', optimizer='adam', metrics=[rmsle])

    
model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/cnn_regression_classify.model"
    
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=7)


model.summary()

tensorcallback = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)

history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[checkpoint, early_stopping, tensorcallback])

test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Validation loss:', test_loss[0])

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



    

