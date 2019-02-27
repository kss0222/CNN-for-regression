
# coding: utf-8

# In[1]:


from PIL import Image
import os, glob, sys, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D, Input, Convolution2D
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
from keras.callbacks import TensorBoard  
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img  
from random import shuffle
from keras.models import Model


img_dir = './dataset'
categories = ['original']
np_classes = len(categories)

image_w = 128
image_h = 128

np.random.seed(0)
input_shape = (1, image_w, image_h) # grayscale 은 channel값 1, colorscale 은 3


X = [] # X는 우리가 알수있는 데이터.
y = [] # X로 예측하고 싶은 데이터. 실제값
filenames = []
for idx, cat in enumerate(categories):
    img_dir_detail = img_dir + "/" + cat
    files = glob.glob(img_dir_detail+"/*/"+ "*.jpg")
    shuffle(files)

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("L") # greyscale (“L”) or “RGB” for color images)
            img = img.resize((image_w, image_h))
            data = np.asarray(img)           
            filenames.append(f)
             
           
            X.append(data)
            y.append(idx)
            
            if i % 100 == 0:
                print(cat, " :\t", filenames[i]+ "  \t", y[i])           
                             
        except:
            print(cat, str(i)+" 번째 에러 ")
            
# normalize data
X = np.array(X)
Y = np.array(y, dtype=np.int64) # 명시적으로 자료타입 알려줌. 


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./numpy_data/binary_image_data2.npy", xy)

X_train, X_test, y_train, y_test = np.load('./numpy_data/binary_image_data2.npy')
# Prepare the training images
X_train = X_train.reshape(X_train.shape[0], image_w, image_h,1)
X_test = X_test.reshape(X_test.shape[0], image_w, image_h,1)

X_train = X_train.astype('float32') / 255 # 흑백 이미지 데이터는 픽셀 하나당 0-255까지의 숫자값 가지므로 255로 나누어 정규화시킴
X_test = X_test.astype('float32') / 255

Y_train = Y_train.astype('float32') / 255
Y_test = Y_test.astype('float32') / 255

print("validation data: {0} \ntest data: {1}".format(X_train.shape, X_test.shape))

input_img = Input(shape=(128, 128, 1))
x = Conv2D(5, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(20, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(25, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

model = Model(input_img, decoded)

def rmse(X_true, X_pred):
        return K.mean(K.square(X_pred - X_true), axis=-1)
adam=Adam(lr=0.00008, beta_1=0.9)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])




model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/con_autoencoder.model"
    
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=7)


model.summary()


tensorcallback = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)

history = model.fit(X_train, X_train,
                    batch_size=16,
                    epochs=1000,
                    verbose=1,
                    validation_data=(X_test, X_test),
                    callbacks=[checkpoint, early_stopping, tensorcallback],
                    shuffle=True)




score = model.evaluate(X_test, X_test, verbose=1)
print(model.metrics_names)
print(score)

