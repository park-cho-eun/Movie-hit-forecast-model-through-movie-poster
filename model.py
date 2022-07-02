from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

import sys
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import load_model

#정형데이터 불러오고 전처리
final = pd.read_csv('\\structed data.csv')

continuous = ['counts', 'R', 'G', 'B']
cs = MinMaxScaler()
Continuous = cs.fit_transform(final[continuous])
Continuous = pd.DataFrame(Continuous, columns=['counts', 'R', 'G', 'B'])

new = final[['인덱스', '레이블']].join(Continuous)
new = new.join(final[['lable', 'director', 'festival', 'award', 'staff']])

from numpy.core.multiarray import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GlobalAveragePooling2D

from keras.applications import vgg16

#vgg-16 모델 로드하고 변형
cnn = vgg16.VGG16(input_shape=(80, 60, 3), include_top= False, weights='imagenet')

global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(4, activation ='softmax')

model = Sequential([cnn, global_average_layer, prediction_layer])

#정형데이터 DNN 모델 생성
input1 = Input(shape=(9,))
x = Dense(8, activation='relu')(input1)
output1 = Dense(4, activation='relu')(x)

#멀티모달 딥러닝
combinedInput = Concatenate()([output1, model.output])
x = Dense(4, activation="relu")(combinedInput)
output = Dense(4, activation="softmax")(x)
model_new = Model(inputs=[input1, model.input], outputs=output)

#Train 이미지 데이터와 정형데이터 인덱스 맞추기
train_data = []

for j in range(4):
  train = os.listdir('/포스터/train/'+str(j))
  for i in train:
    i = i.split('n')[1]
    i = i.split('.')[0]
    train_data.append(int(i))

trainAttr = pd.DataFrame([])

for k in train_data:
  trainAttr = pd.concat([trainAttr, new[new['인덱스']==k]])

trainAttr = trainAttr[trainAttr['인덱스']<=733]
trainAttr = trainAttr.sort_values(by=['인덱스'])
trainAttrX = trainAttr[['counts', 'R', 'G', 'B', 'lable', 'director', 'festival', 'award', 'staff']]
trainAttrY = trainAttr[['레이블']]

#Test 이미지 데이터와 정형데이터 인덱스 맞추기
test_data = []

for j in range(4):
  test = os.listdir('/포스터/test/'+str(j))
  for i in test:
    i = i.split('n')[1]
    i = i.split('.')[0]
    test_data.append(int(i))

testAttr = pd.DataFrame([])

for k in test_data:
  testAttr = pd.concat([testAttr, new[new['인덱스']==k]])

testAttr = testAttr[testAttr['인덱스']<=727]
testAttr = testAttr.sort_values(by=['인덱스'])
testAttrX = testAttr[['counts', 'R', 'G', 'B', 'lable', 'director', 'festival', 'award', 'staff']]
testAttrY = testAttr[['레이블']]

#one-hot encoding
from tensorflow.keras.utils import to_categorical
trainAttrY = to_categorical(trainAttrY)
testAttrY = to_categorical(testAttrY)

#train 이미지 데이터셋 불러오기
train_images = []

for i in sorted(list(trainAttr['인덱스'])):
  image = cv2.imread('/포스터/train/all/train'+str(i)+'.jpg의 사본')
  if image is None:
    pass
  else:
    image = cv2.resize(image, (60, 80))
    train_images.append(image)

train_images = np.array(train_images)

#test 이미지 데이터셋 불러오기
test_images = []

for i in sorted(list(testAttr['인덱스'])):
  image = cv2.imread('/포스터/test/all/train'+str(i)+'.jpg의 사본')
  if image is None:
    pass
  else:
    image = cv2.resize(image, (60, 80))
    test_images.append(image)

test_images = np.array(test_images)

#모델 학습
from keras.callbacks import EarlyStopping
model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'])
early_stopping = EarlyStopping(patience = 3)
model_new.fit([trainAttrX, train_images], trainAttrY, callbacks = [early_stopping],
                     validation_data=([testAttrX, test_images], testAttrY), epochs=100)

#모델 평가
from sklearn.metrics import f1_score

predict = model_new.predict([testAttrX, test_images])
predict = np.argmax(np.squeeze(predict), axis=1)

f1_score(testAttr[['레이블']], predict, average='weighted')
model_new.evaluate([testAttrX, test_images], testAttrY)