
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

predictions = model.predict(X_test, batch_size=64, verbose=1)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("mean_squared_error")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()


# In[2]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("mean_squared_error")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()


# In[ ]:



#--------------------
from PIL import Image
from keras.models import load_model
import os, glob, sys, numpy as np
import matplotlib.pyplot as plt

#load test image


#load test image

caltech_dir = './dataset/test'


image_w = 128
image_h = 128


np.random.seed(7)


X = []
y = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("L")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)
    
    file_token=filenames[i].split("_")[1]  
    
    


X = np.array(X)
X = X.reshape(X.shape[0], image_w, image_h, 1)
X = X.astype('float32') / 255

model = load_model('./model/con_autoencoder.model')


# make a prediction

pred = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

n=10
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    ax=plt.subplot(2, n, i+1)
    plt.imshow(X[i].reshape(128,128), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")

for i in range(10):
    ax=plt.subplot(2, n, i+1)
    plt.imshow(pred[i].reshape(128,128), cmap='gray')
    print("해당 " + filenames[i] + " \t"+  "  이미지의 y값은 " + str(np.mean((X[i]-pred[i])*(X[i]-pred[i]))) + "값으로 예측됩니다.")
    print(np.mean((X[i]-pred[i])*(X[i]-pred[i])))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[2]:



import glob
import os
# 현재 위치의 파일 목록 

dirpath = './dataset/test_token'



#디렉토리안의 mp3확장자를 가지고 있는 파일들을 찾아준다.
file_list = glob.glob("./dataset/test_token/*.jpg") 



count = 0

for name in file_list:
    
    #카운트할 숫자
    count = count + 1
    
    #슬라이싱한 스트링에 숫자를 새로 정의하고 기존의 이름에서 남길 부분을 붙여준다.
    name_change = "./dataset/test_token/"+str(count)+".jpg" 
    print(name_change)
    #파일이름을 변환시킨다.
    os.rename(name, name_change)

 



        


# In[52]:


from sklearn.preprocessing import normalize
data = np.array([
    [1000, 10, 0.5],
    [765, 5, 0.35],
    [800, 7, 0.09], ])
data = normalize(data, axis=0, norm='max')
print(data)


# In[1]:


import numpy as np
test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

print(normalize(test_array))    
# [ 0.     0.125  0.25   0.375  0.5    0.625  0.75   0.875  1.   ]


# In[1]:


import sys
import numpy as np
import pandas as pd

input_file = 'Vol_Comp.csv'
output_file = 'data_output1.csv'

data_frame = pd.read_csv(input_file)
#data_frame['Novelty'] = data_frame['Novelty'].astype(float)


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x)) # np.max(x)-np.min(x)
    
print(normalize(data_frame['Compliance']))   
data_frame['Volume_Nor'] = normalize(data_frame['Volume'])
data_frame['Compliance_Nor'] = normalize(data_frame['Compliance'])
data_frame['Novelty_Nor'] = normalize(data_frame['Novelty'])
data_frame.to_csv(output_file, index=False)


# In[2]:


import sys
import numpy as np
import pandas as pd

input_file = 'Vol_Comp.csv'
output_file = 'data_output2.csv'

data_frame = pd.read_csv(input_file)
#data_frame['Novelty'] = data_frame['Novelty'].astype(float)


def standardization(x):
    x = np.asarray(x)
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0)) #N(0,1) 평균=0, 표준편차 =1 
    
print(standardization(data_frame['Compliance']))   

data_frame['Volume_Std'] = standardization(data_frame['Volume'])
data_frame['Compliance_Std'] = standardization(data_frame['Compliance'])
data_frame['Novelty_Std'] = standardization(data_frame['Novelty'])
data_frame.to_csv(output_file, index=False)

